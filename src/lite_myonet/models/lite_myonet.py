import math
from numbers import Integral
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from lite_myonet.data.bones import joints_to_bones
from .skeleton import SkeletonDefinition


STUDENT_FEATURE_KEYS = (
    "joint_stage1",
    "bone_vector_stage1",
    "joint_stage2",
    "bone_vector_stage2",
    "joint_stage3",
    "bone_vector_stage3",
)


def _validate_positive_integer(name: str, value: int) -> int:
    """Validate a positive integer model setting."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def build_physical_adjacency(
    num_points: int,
    skeleton_edges: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Build a symmetric degree-normalized adjacency matrix with self loops."""
    num_points = _validate_positive_integer("num_points", num_points)
    if not isinstance(skeleton_edges, Sequence) or isinstance(skeleton_edges, (str, bytes)) or not skeleton_edges:
        raise ValueError("skeleton_edges must be a non-empty edge sequence.")

    adjacency = torch.eye(num_points, dtype=torch.float32)
    seen = set()
    for edge in skeleton_edges:
        if not isinstance(edge, Sequence) or len(edge) != 2:
            raise ValueError("Each skeleton edge must contain two point indices.")
        left, right = edge
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not isinstance(left, Integral)
            or not isinstance(right, Integral)
        ):
            raise ValueError("Skeleton point indices must be integers.")
        left, right = int(left), int(right)
        if left == right or not (0 <= left < num_points and 0 <= right < num_points):
            raise ValueError(f"Invalid skeleton edge ({left}, {right}).")
        edge_key = (min(left, right), max(left, right))
        if edge_key in seen:
            raise ValueError(f"Duplicate skeleton edge: {edge_key}.")
        seen.add(edge_key)
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0

    degree_inverse_sqrt = adjacency.sum(dim=1).clamp_min(1.0).rsqrt()
    adjacency = degree_inverse_sqrt[:, None] * adjacency * degree_inverse_sqrt[None, :]
    return adjacency.contiguous()


class StaticDynamicGraphConvolution(nn.Module):
    """Combine physical and frame-wise learned graph aggregation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
    ):
        """Initialize fixed-graph and input-dependent graph aggregation."""
        super().__init__()
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be a square two-dimensional tensor.")
        if not torch.is_floating_point(adjacency):
            raise ValueError("adjacency must be a floating-point tensor.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_points = int(adjacency.shape[0])
        self.feature_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.adjacency_bias = nn.Parameter(torch.zeros_like(adjacency))
        self.query_channels = max(1, in_channels // 4)
        self.query_projection = nn.Conv2d(in_channels, self.query_channels, kernel_size=1)
        self.key_projection = nn.Conv2d(in_channels, self.query_channels, kernel_size=1)
        self.scale = math.sqrt(self.query_channels)
        self.output_normalization = nn.BatchNorm2d(out_channels)
        self.register_buffer(
            "adjacency", adjacency.detach().clone().float(), persistent=False
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Aggregate node features through both graph paths."""
        batch_size, _, frame_count, point_count = features.shape
        if point_count != self.num_points:
            raise ValueError(
                f"Expected {self.num_points} graph points, received {point_count}."
            )

        projected = self.feature_projection(features)
        output = torch.matmul(projected, self.adjacency + self.adjacency_bias)
        query = self.query_projection(features).permute(0, 2, 3, 1)
        query = query.reshape(batch_size * frame_count, point_count, self.query_channels)
        key = self.key_projection(features).permute(0, 2, 1, 3)
        key = key.reshape(batch_size * frame_count, self.query_channels, point_count)
        dynamic_adjacency = torch.softmax(torch.matmul(query, key) / self.scale, dim=-1)
        dynamic_adjacency = dynamic_adjacency.reshape(
            batch_size, frame_count, point_count, point_count
        )
        dynamic_output = torch.matmul(
            projected.permute(0, 2, 1, 3), dynamic_adjacency
        ).permute(0, 2, 1, 3)
        return self.output_normalization(output + dynamic_output)


class FourierTemporalConvolution(nn.Module):
    """Apply a depthwise temporal filter and a Fourier channel projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        fourier_degree: int = 2,
    ):
        """Initialize a depthwise temporal convolution and Fourier channel mapping."""
        super().__init__()
        in_channels = _validate_positive_integer("in_channels", in_channels)
        out_channels = _validate_positive_integer("out_channels", out_channels)
        stride = _validate_positive_integer("stride", stride)
        if (
            isinstance(fourier_degree, bool)
            or not isinstance(fourier_degree, Integral)
            or int(fourier_degree) not in (1, 2, 3)
        ):
            raise ValueError("fourier_degree must be 1, 2, or 3.")
        fourier_degree = int(fourier_degree)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fourier_degree = fourier_degree
        self.temporal_depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 1),
            stride=(stride, 1),
            padding=(1, 0),
            groups=in_channels,
            bias=False,
        )
        self.depthwise_normalization = nn.BatchNorm2d(in_channels)
        self.trend_convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.trend_normalization = nn.BatchNorm2d(out_channels)
        self.trend_activation = nn.ReLU(inplace=False)
        self.output_normalization = nn.BatchNorm2d(out_channels)
        self.fourier_coefficients = nn.Parameter(
            torch.randn(fourier_degree, 2, in_channels, out_channels)
            / math.sqrt(in_channels)
        )

    def _fourier_projection(self, features: torch.Tensor) -> torch.Tensor:
        """Project each temporal feature through paired Fourier harmonics."""
        channel_last = features.permute(0, 2, 3, 1)
        output = channel_last.new_zeros(
            channel_last.shape[:-1] + (self.out_channels,)
        )
        for degree_index in range(self.fourier_degree):
            harmonic = degree_index + 1
            cosine_features = torch.cos(harmonic * channel_last)
            sine_features = torch.sin(harmonic * channel_last)
            output = output + torch.matmul(
                cosine_features, self.fourier_coefficients[degree_index, 0]
            )
            output = output + torch.matmul(
                sine_features, self.fourier_coefficients[degree_index, 1]
            )
        return output.permute(0, 3, 1, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the sum of Fourier and trend responses."""
        basis_features = self.depthwise_normalization(
            self.temporal_depthwise(features)
        )
        trend = self.trend_activation(
            self.trend_normalization(self.trend_convolution(basis_features))
        )
        nonlinear = self._fourier_projection(basis_features)
        return self.output_normalization(trend + nonlinear)


class StudentGraphTemporalBlock(nn.Module):
    """Combine learned graph aggregation with Fourier temporal projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        stride: int,
        fourier_degree: int,
    ):
        """Initialize a residual graph-Fourier block at one student stage."""
        super().__init__()
        self.graph_convolution = StaticDynamicGraphConvolution(
            in_channels,
            out_channels,
            adjacency,
        )
        self.temporal_convolution = FourierTemporalConvolution(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=stride,
            fourier_degree=fourier_degree,
        )
        if stride == 1 and in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return a graph-temporal feature map with a residual connection."""
        residual = self.residual(features)
        output = self.temporal_convolution(self.graph_convolution(features))
        return F.relu(output + residual, inplace=False)


class ChannelwiseStreamFusion(nn.Module):
    """Fuse joint and bone features with a learned channel gate."""

    def __init__(self, channels: int):
        """Initialize channelwise gates for joint and bone feature fusion."""
        super().__init__()
        hidden_channels = max(1, channels // 2)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 2, hidden_channels),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_channels, channels * 2),
            nn.Sigmoid(),
        )

    def forward(self, joint_features: torch.Tensor, bone_features: torch.Tensor) -> torch.Tensor:
        """Fuse two feature maps using complementary channel weights."""
        if joint_features.shape != bone_features.shape:
            raise ValueError("Joint and bone feature maps must have the same shape.")
        weights = self.gate(torch.cat((joint_features, bone_features), dim=1))
        joint_weight, bone_weight = weights.chunk(2, dim=1)
        joint_weight = joint_weight[:, :, None, None]
        bone_weight = bone_weight[:, :, None, None]
        return joint_features * joint_weight + bone_features * bone_weight


class CosineClassifierHead(nn.Module):
    """Classify normalized features with cosine similarity."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float):
        """Initialize normalized class weights, learnable scale, and dropout."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.weight = nn.Parameter(torch.empty(num_classes, in_channels))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        self.log_scale = nn.Parameter(torch.tensor(math.log(math.expm1(9.0))))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return cosine logits with a learned positive scale."""
        pooled = self.dropout(self.pool(features).flatten(1))
        normalized_features = F.normalize(pooled, p=2, dim=1)
        normalized_weights = F.normalize(self.weight, p=2, dim=1)
        scale = 1.0 + F.softplus(self.log_scale)
        return scale * (normalized_features @ normalized_weights.t())


class LiteMyoNet(nn.Module):
    """Two-stream student with differentiable bones and Fourier temporal layers."""

    feature_keys = STUDENT_FEATURE_KEYS

    def __init__(
        self,
        skeleton: SkeletonDefinition,
        num_classes: int = 33,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128),
        dropout: float = 0.05,
        fourier_degree: int = 2,
    ):
        """Build the complete two-stream Fourier student from a skeleton definition."""
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        self.skeleton = skeleton
        self.num_classes = _validate_positive_integer("num_classes", num_classes)
        self.num_points = skeleton.point_count
        self.in_channels = _validate_positive_integer("in_channels", in_channels)
        if isinstance(channels, (str, bytes)) or not isinstance(channels, Sequence) or len(channels) != 3:
            raise ValueError("channels must contain three positive stage widths.")
        self.channels = tuple(
            _validate_positive_integer(f"channels[{index}]", value)
            for index, value in enumerate(channels)
        )
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise ValueError("dropout must be a finite value in [0, 1).")
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be a finite value in [0, 1).")
        self.dropout_rate = float(dropout)
        self.fourier_degree = _validate_positive_integer("fourier_degree", fourier_degree)
        if self.fourier_degree not in (1, 2, 3):
            raise ValueError("fourier_degree must be 1, 2, or 3.")

        adjacency = build_physical_adjacency(
            self.num_points, skeleton.model_graph_edges
        )
        self.register_buffer("physical_adjacency", adjacency, persistent=False)

        stage_strides = (2, 2, 1)
        stage_channels = (self.in_channels,) + self.channels
        self.joint_stream = nn.ModuleList()
        self.bone_vector_stream = nn.ModuleList()
        for stage_index, stride in enumerate(stage_strides):
            in_stage_channels = stage_channels[stage_index]
            out_stage_channels = self.channels[stage_index]
            self.joint_stream.append(
                StudentGraphTemporalBlock(
                    in_channels=in_stage_channels,
                    out_channels=out_stage_channels,
                    adjacency=adjacency,
                    stride=stride,
                    fourier_degree=self.fourier_degree,
                )
            )
            self.bone_vector_stream.append(
                StudentGraphTemporalBlock(
                    in_channels=in_stage_channels,
                    out_channels=out_stage_channels,
                    adjacency=adjacency,
                    stride=stride,
                    fourier_degree=self.fourier_degree,
                )
            )

        self.fusion = ChannelwiseStreamFusion(self.channels[-1])
        self.classifier = CosineClassifierHead(
            self.channels[-1], self.num_classes, self.dropout_rate
        )

    def count_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _validate_input(self, joints: torch.Tensor) -> None:
        """Validate NCTV coordinates against this model's skeleton."""
        if not isinstance(joints, torch.Tensor):
            raise TypeError("joints must be a torch.Tensor.")
        if joints.ndim != 4:
            raise ValueError(f"joints must have shape [N,C,T,V], got {tuple(joints.shape)}.")
        if joints.shape[0] <= 0 or joints.shape[2] <= 0:
            raise ValueError("Batch size and frame count must be positive.")
        if joints.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {joints.shape[1]}.")
        if joints.shape[3] != self.num_points:
            raise ValueError(f"Expected {self.num_points} skeleton points, got {joints.shape[3]}.")
        if not torch.is_floating_point(joints):
            raise TypeError("joints must use a floating-point dtype.")
        if joints.device != self.classifier.weight.device:
            raise ValueError("Input and model must be on the same device.")
        if joints.dtype != self.classifier.weight.dtype:
            raise TypeError("Input and model must use the same dtype.")

    def forward(self, joints: torch.Tensor, return_features: bool = False):
        """Return logits and, when requested, the six stage feature maps."""
        if not isinstance(return_features, bool):
            raise TypeError("return_features must be a bool.")
        self._validate_input(joints)
        bone_vectors = joints_to_bones(joints, self.skeleton)

        joint_features = joints
        bone_features = bone_vectors
        captured = {}
        for stage_index, (joint_block, bone_block) in enumerate(
            zip(self.joint_stream, self.bone_vector_stream), start=1
        ):
            joint_features = joint_block(joint_features)
            bone_features = bone_block(bone_features)
            if return_features:
                captured[f"joint_stage{stage_index}"] = joint_features
                captured[f"bone_vector_stage{stage_index}"] = bone_features

        fused_features = self.fusion(joint_features, bone_features)
        logits = self.classifier(fused_features)
        if return_features:
            return logits, {name: captured[name] for name in STUDENT_FEATURE_KEYS}
        return logits
