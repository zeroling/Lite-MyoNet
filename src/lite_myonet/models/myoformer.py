import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lite_myonet.data.bones import joints_to_bones
from .bone_encoder import MultiScaleBoneTemporalBlock
from .skeleton import SkeletonDefinition
from .attention.structure_aware_spatial import StructureAwareSpatialAttention
from .attention.symmetry_aware_temporal import SymmetryAwareTemporalAttention


def _build_partitioned_adjacency(skeleton: SkeletonDefinition) -> torch.Tensor:
    """Build the root-relative three-part adjacency used by the teacher streams."""
    point_count = skeleton.point_count
    edges = skeleton.model_graph_edges
    adjacency = torch.zeros((3, point_count, point_count), dtype=torch.float32)
    adjacency[0] = torch.eye(point_count)
    graph = torch.zeros((point_count, point_count), dtype=torch.float32)
    if edges:
        edge_tensor = torch.tensor(edges, dtype=torch.long)
        left, right = edge_tensor[:, 0], edge_tensor[:, 1]
        graph[left, right] = 1.0
        graph[right, left] = 1.0
        distances = torch.full((point_count,), float("inf"))
        distances[skeleton.root_index] = 0.0
        queue = [skeleton.root_index]
        while queue:
            node = queue.pop(0)
            neighbors = (graph[node] > 0).nonzero(as_tuple=False).flatten().tolist()
            for neighbor in neighbors:
                if torch.isinf(distances[neighbor]):
                    distances[neighbor] = distances[node] + 1.0
                    queue.append(neighbor)
        distances = distances.long()
        left_distance, right_distance = distances[left], distances[right]
        close_mask = left_distance <= right_distance
        adjacency[1, left[close_mask], right[close_mask]] = 1.0
        reverse_close = left_distance > right_distance
        adjacency[1, right[reverse_close], left[reverse_close]] = 1.0
        far_mask = left_distance > right_distance
        adjacency[2, left[far_mask], right[far_mask]] = 1.0
        reverse_far = left_distance < right_distance
        adjacency[2, right[reverse_far], left[reverse_far]] = 1.0
    degree = adjacency.sum(dim=2, keepdim=True)
    degree[degree == 0] = 1.0
    return (adjacency / degree).contiguous()


class GraphConvolution(nn.Module):
    """Apply a learned channel projection and the supplied spatial adjacency."""
    def __init__(self, in_channels, out_channels, kernel_size, A, bias=True):
        """Initialize a graph convolution with a fixed partitioned adjacency."""
        super(GraphConvolution, self).__init__()
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=1,
            bias=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


        self.register_buffer(
            'A', torch.as_tensor(A, dtype=torch.float32).detach().clone(), persistent=False
        )  # [K, V, V]

    def reset_parameters(self):
        """Initialize convolution weights and the optional output bias."""
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        """Project channels and aggregate each graph-adjacency partition."""
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A))
        if self.bias is not None:
            return x + self.bias.view(1, -1, 1, 1)
        else:
            return x

class JointTemporalAttentionBlock(nn.Module):
    """Apply pre-activated graph convolution and symmetry-aware temporal attention."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dropout,
        adjacency,
        attention_config: Optional[dict] = None,
        symmetry_pairs=None,
    ):
        """Initialize a residual graph and symmetry-aware temporal block."""
        super().__init__()
        self.stride = stride
        self.dropout = dropout
        self.attention_config = dict(attention_config or {})
        if symmetry_pairs is not None:
            self.attention_config.setdefault("symmetry_pairs", symmetry_pairs)

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.gcn_pre = nn.Sequential(
            self.norm1,
            nn.ReLU(inplace=True),
            GraphConvolution(in_channels, out_channels, kernel_size[0], adjacency),
            self.norm2,
            nn.ReLU(inplace=True),
        )
        self.temporal_attn = SymmetryAwareTemporalAttention(
            out_channels, dropout_rate=dropout, **self.attention_config
        )

        if stride == 2:
            self.temporal_conv = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size[1], 1),
                stride=(stride, 1),
                padding=(kernel_size[1] // 2, 0),
            )
            nn.init.kaiming_normal_(self.temporal_conv.weight, mode="fan_out", nonlinearity="relu")
            if self.temporal_conv.bias is not None:
                nn.init.constant_(self.temporal_conv.bias, 0)
        else:
            self.temporal_conv = nn.Identity()

        if stride == 2 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Sequential()

    def forward(self, features):
        """Return a residual graph-temporal representation."""
        residual = self.residual(features)
        output = self.temporal_attn(self.gcn_pre(features))
        output = self.temporal_conv(output)
        output = output + residual
        return F.dropout(output, self.dropout, self.training)


class JointSpatialAttentionBlock(nn.Module):
    """Apply structure-aware spatial attention before a pre-activated temporal block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dropout,
        num_points,
        bones,
        attention_config: Optional[dict] = None,
    ):
        """Initialize a residual structure-aware spatial-attention block."""
        super().__init__()
        self.dropout = dropout
        self.attention_config = dict(attention_config or {})
        self.spatial_attn = StructureAwareSpatialAttention(
            in_channels,
            num_points=num_points,
            bones=bones,
            dropout_rate=dropout,
            **self.attention_config,
        )
        self.norm_in = nn.BatchNorm2d(in_channels)
        self.norm_out = nn.BatchNorm2d(out_channels)
        self.tcn_conv = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_size[1], 1),
            (stride, 1),
            padding=((kernel_size[1] - 1) // 2, 0),
        )
        nn.init.kaiming_normal_(self.tcn_conv.weight, mode="fan_out", nonlinearity="relu")
        if self.tcn_conv.bias is not None:
            nn.init.constant_(self.tcn_conv.bias, 0)

        if in_channels == out_channels and stride == 1:
            self.residual = nn.Sequential()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, features):
        """Return a residual spatial-temporal representation."""
        residual = self.residual(features)
        output = self.spatial_attn(features)
        output = self.norm_in(output)
        output = F.relu(output, inplace=True)
        output = self.tcn_conv(output)
        output = self.norm_out(output)
        output = output + residual
        return F.dropout(output, self.dropout, self.training)


class SpatioTemporalGraphBlock(nn.Module):
    """Apply normalized graph convolution followed by temporal convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout, adjacency):
        """Initialize a residual graph and temporal convolution block."""
        super().__init__()
        self.gcn_norm = nn.BatchNorm2d(in_channels)
        self.tcn_norm = nn.BatchNorm2d(out_channels)
        self.gcn = nn.Sequential(
            self.gcn_norm,
            nn.ReLU(inplace=True),
            GraphConvolution(in_channels, out_channels, kernel_size[0], adjacency),
        )
        self.tcn_conv = nn.Conv2d(
            out_channels,
            out_channels,
            (kernel_size[1], 1),
            (stride, 1),
            padding=((kernel_size[1] - 1) // 2, 0),
        )
        nn.init.kaiming_normal_(self.tcn_conv.weight, mode="fan_out", nonlinearity="relu")
        if self.tcn_conv.bias is not None:
            nn.init.constant_(self.tcn_conv.bias, 0)

        if in_channels == out_channels and stride == 1:
            self.residual = nn.Sequential()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.dropout = dropout

    def forward(self, features):
        """Return a residual spatio-temporal graph representation."""
        residual = self.residual(features)
        output = self.gcn(features)
        output = self.tcn_norm(output)
        output = F.relu(output, inplace=True)
        output = self.tcn_conv(output)
        output = F.dropout(output, self.dropout, self.training)
        return output + residual


class CosineClassifierHead(nn.Module):
    """Classify pooled representations using scaled normalized class weights."""
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.3,
        scale_init: float = 10.0,
        scale_min: float = 1.0,
        scale_max: float = 30.0,
        norm: str = "layernorm",
        num_mc_samples: int = 0,
        mc_dropout_eval: bool = False,
    ):
        """Initialize normalized class weights, learnable scale, and dropout."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.num_mc_samples = int(num_mc_samples) if num_mc_samples is not None else 0
        self.mc_dropout_eval = bool(mc_dropout_eval)
        if self.num_mc_samples < 0:
            raise ValueError("num_mc_samples must be non-negative.")
        if scale_min < 0:
            raise ValueError("scale_min must be non-negative.")
        if scale_init < scale_min:
            raise ValueError("scale_init must be at least scale_min.")
        if scale_max is not None and scale_max <= scale_min:
            raise ValueError("scale_max must be greater than scale_min.")
        if scale_max is not None and scale_init > scale_max:
            raise ValueError("scale_init must not exceed scale_max.")


        norm = (norm or "").lower()
        if norm == "layernorm":
            self.norm = nn.LayerNorm(in_channels)
        elif norm == "batchnorm":
            self.norm = nn.BatchNorm1d(in_channels)
        else:
            self.norm = None


        self.weight = nn.Parameter(torch.empty(num_classes, in_channels))
        nn.init.kaiming_normal_(self.weight, nonlinearity='linear')


        scale_increment = max(1e-6, float(scale_init) - float(scale_min))
        raw_scale = math.log(math.expm1(scale_increment))
        self.log_scale = nn.Parameter(torch.tensor(raw_scale, dtype=torch.float32))
        self.register_buffer("scale_min", torch.tensor(float(scale_min)))
        self.register_buffer("scale_max", torch.tensor(float(scale_max if scale_max is not None else 0.0)))
        self.has_scale_max = scale_max is not None and float(scale_max) > 0.0

    def _compute_logits(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, C]
        """Compute scaled cosine similarities to normalized class weights."""
        z = F.normalize(z, dim=1)
        w = F.normalize(self.weight, dim=1)
        s = self.scale_min + F.softplus(self.log_scale)
        if self.has_scale_max:
            s = torch.clamp(s, max=self.scale_max)
        return s * (z @ w.t())

    def _apply_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Apply training dropout or explicitly enabled Monte Carlo dropout."""
        return F.dropout(
            x,
            p=self.dropout.p,
            training=self.training or self.mc_dropout_eval,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """Pool features and return normalized cosine-classifier logits."""
        x = self.pool(x)
        x = self.flatten(x)  # [B, C]
        if self.norm is not None:
            x = self.norm(x)

        mc = self.num_mc_samples if (self.training or self.mc_dropout_eval) else 0
        if mc and mc > 1:
            logits = 0.0
            for _ in range(mc):
                logits = logits + self._compute_logits(self._apply_dropout(x))
            logits = logits / float(mc)
        else:
            logits = self._compute_logits(self._apply_dropout(x))
        return logits

class MyoFormer(nn.Module):
    """Three-stream teacher with a shared cosine classifier."""

    def __init__(
        self,
        skeleton: SkeletonDefinition,
        num_classes=33,
        in_channels=3,
        dropout=0.05,
        spatial_attention_config: Optional[dict] = None,
        symmetry_attention_config: Optional[dict] = None,
        scale_attention_config: Optional[dict] = None,
        head_config: Optional[dict] = None,
    ):
        """Build the complete three-stream teacher from a skeleton definition."""
        super().__init__()
        self.skeleton = skeleton
        self.num_points = skeleton.point_count
        self.in_channels = in_channels
        self.bones = skeleton.model_graph_edges
        self.symmetry_pairs = tuple(skeleton.symmetry_pairs)
        self.register_buffer("A", _build_partitioned_adjacency(skeleton), persistent=False)
        self.num_classes = num_classes

        self.spatial_attention_config = dict(spatial_attention_config or {})
        self.symmetry_attention_config = dict(symmetry_attention_config or {})
        self.scale_attention_config = dict(scale_attention_config or {})

        adjacency = self.A
        self.joint_spatial_stream = nn.ModuleList([
            SpatioTemporalGraphBlock(in_channels, 64, (3, 3), 1, dropout, adjacency),
            SpatioTemporalGraphBlock(64, 64, (3, 3), 2, dropout, adjacency),
            SpatioTemporalGraphBlock(64, 64, (3, 3), 1, dropout, adjacency),
            JointSpatialAttentionBlock(
                64, 128, (3, 3), 1, dropout, self.num_points, self.bones,
                attention_config=self.spatial_attention_config,
            ),
            JointSpatialAttentionBlock(
                128, 128, (3, 3), 1, dropout, self.num_points, self.bones,
                attention_config=self.spatial_attention_config,
            ),
            JointSpatialAttentionBlock(
                128, 128, (3, 3), 1, dropout, self.num_points, self.bones,
                attention_config=self.spatial_attention_config,
            ),
            JointSpatialAttentionBlock(
                128, 256, (3, 3), 2, dropout, self.num_points, self.bones,
                attention_config=self.spatial_attention_config,
            ),
            JointSpatialAttentionBlock(
                256, 256, (3, 3), 1, dropout, self.num_points, self.bones,
                attention_config=self.spatial_attention_config,
            ),
            JointSpatialAttentionBlock(
                256, 256, (3, 3), 2, dropout, self.num_points, self.bones,
                attention_config=self.spatial_attention_config,
            ),
        ])

        temporal_stages = (
            (64, 128, 1),
            (128, 128, 1),
            (128, 128, 1),
            (128, 256, 2),
            (256, 256, 1),
            (256, 256, 2),
        )
        self.joint_temporal_stream = nn.ModuleList([
            SpatioTemporalGraphBlock(in_channels, 64, (3, 3), 1, dropout, adjacency),
            SpatioTemporalGraphBlock(64, 64, (3, 3), 2, dropout, adjacency),
            SpatioTemporalGraphBlock(64, 64, (3, 3), 1, dropout, adjacency),
            *[
                JointTemporalAttentionBlock(
                    input_channels,
                    output_channels,
                    (3, 3),
                    stride,
                    dropout,
                    adjacency,
                    attention_config=self.symmetry_attention_config,
                    symmetry_pairs=self.symmetry_pairs,
                )
                for input_channels, output_channels, stride in temporal_stages
            ],
        ])

        self.bone_vector_stream = nn.ModuleList([
            MultiScaleBoneTemporalBlock(in_channels, 64, dropout),
            MultiScaleBoneTemporalBlock(64, 64, dropout, time_stride=2),
            MultiScaleBoneTemporalBlock(
                64, 64, dropout, use_attention=True,
                attention_config=self.scale_attention_config,
            ),
            MultiScaleBoneTemporalBlock(
                64, 128, dropout, use_attention=True,
                attention_config=self.scale_attention_config,
            ),
            MultiScaleBoneTemporalBlock(
                128, 128, dropout, use_attention=True,
                attention_config=self.scale_attention_config,
            ),
            MultiScaleBoneTemporalBlock(
                128, 128, dropout, use_attention=True,
                attention_config=self.scale_attention_config,
            ),
            MultiScaleBoneTemporalBlock(
                128, 256, dropout, time_stride=2, use_attention=True,
                attention_config=self.scale_attention_config,
            ),
            MultiScaleBoneTemporalBlock(
                256, 256, dropout, use_attention=True,
                attention_config=self.scale_attention_config,
            ),
            MultiScaleBoneTemporalBlock(
                256, 256, dropout, time_stride=2, use_attention=True,
                attention_config=self.scale_attention_config,
            ),
        ])

        hcfg = dict(head_config or {})
        self.classifier = CosineClassifierHead(
            in_channels=256,
            num_classes=num_classes,
            dropout=hcfg.get("dropout", 0.30),
            scale_init=hcfg.get("scale_init", 10.0),
            scale_min=hcfg.get("scale_min", 1.0),
            scale_max=hcfg.get("scale_max", 20.0),
            norm=hcfg.get("norm", "layernorm"),
            num_mc_samples=int(hcfg.get("num_mc_samples", 0)),
            mc_dropout_eval=bool(hcfg.get("mc_dropout_eval", False)),
        )

    @staticmethod
    def _forward_stream(features, layers, feature_indices=()):
        """Apply one stream and optionally capture selected intermediate features."""
        captured = []
        capture_indices = set(feature_indices)
        for index, layer in enumerate(layers):
            features = layer(features)
            if index in capture_indices:
                captured.append(features)
        return features, captured

    def count_params(self):
        """Return the number of trainable parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _validate_input(self, joints: torch.Tensor) -> None:
        """Validate NCTV joint coordinates against the skeleton contract."""
        if not isinstance(joints, torch.Tensor):
            raise TypeError("joints must be a torch.Tensor.")
        if joints.ndim != 4:
            raise ValueError(f"joints must have shape [N,C,T,V], got {tuple(joints.shape)}.")
        if joints.shape[0] <= 0 or joints.shape[2] <= 0:
            raise ValueError("Batch size and frame count must be positive.")
        if joints.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {joints.shape[1]}.")
        if joints.shape[3] != self.num_points:
            raise ValueError(f"Expected {self.num_points} skeleton points, got {joints.shape[3]}.")
        if not torch.is_floating_point(joints):
            raise TypeError("joints must use a floating-point dtype.")

    def forward(self, joints: torch.Tensor, return_features=False):
        """Return logits, optionally with the nine stream features used for distillation."""
        self._validate_input(joints)
        bone_vectors = joints_to_bones(joints, self.skeleton)
        feature_indices = (3, 6, 8) if return_features else ()
        spatial, spatial_features = self._forward_stream(
            joints, self.joint_spatial_stream, feature_indices
        )
        temporal, temporal_features = self._forward_stream(
            joints, self.joint_temporal_stream, feature_indices
        )
        bone, bone_features = self._forward_stream(
            bone_vectors, self.bone_vector_stream, feature_indices
        )
        logits = self.classifier(spatial + temporal + bone)
        if return_features:
            return logits, spatial_features + temporal_features + bone_features
        return logits
