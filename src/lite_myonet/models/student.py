import math
from numbers import Integral
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import topology


STUDENT_FEATURE_KEYS = (
    "joint_stage1",
    "bone_vector_stage1",
    "joint_stage2",
    "bone_vector_stage2",
    "joint_stage3",
    "bone_vector_stage3",
)

_TEMPORAL_BASES = frozenset(("standard", "fourier", "bspline"))


def _validate_positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} 必须是正整数。")
    return int(value)


def build_physical_adjacency(
    num_points: int,
    skeleton_edges: Optional[Sequence[Tuple[int, int]]] = None,
) -> torch.Tensor:
    """从 22 条无向骨架边构造论文所定义的原始邻接矩阵，无自环、无度归一化。"""
    num_points = _validate_positive_integer("num_points", num_points)
    edges = topology.bones if skeleton_edges is None else skeleton_edges
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)) or not edges:
        raise ValueError("skeleton_edges 必须是非空边序列。")

    adjacency = torch.zeros(num_points, num_points, dtype=torch.float32)
    normalized_edges = set()
    for edge in edges:
        if not isinstance(edge, Sequence) or len(edge) != 2:
            raise ValueError("每条 skeleton edge 必须包含两个关节索引。")
        source, target = edge
        if (
            isinstance(source, bool)
            or isinstance(target, bool)
            or not isinstance(source, Integral)
            or not isinstance(target, Integral)
        ):
            raise ValueError("骨架关节索引必须是整数。")
        source, target = int(source), int(target)
        if source == target:
            raise ValueError("skeleton_edges 不应包含自连接。")
        if not (0 <= source < num_points and 0 <= target < num_points):
            raise ValueError(
                f"骨架关节索引 ({source}, {target}) 超出 num_points={num_points}。"
            )
        edge_key = (min(source, target), max(source, target))
        if edge_key in normalized_edges:
            raise ValueError(f"skeleton_edges 中存在重复边：{edge_key}。")
        normalized_edges.add(edge_key)
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0

    return adjacency


def build_spatial_partitions(num_points, skeleton_edges=None):
    """[I, inward, outward], normalized by destination degree for X @ A."""
    edges = topology.bones if skeleton_edges is None else skeleton_edges
    build_physical_adjacency(num_points, edges)
    inward = torch.zeros(num_points, num_points, dtype=torch.float32)
    for child, parent in edges:
        inward[child, parent] = 1.0
    outward = inward.t().contiguous()
    partitions = torch.stack((torch.eye(num_points), inward, outward))
    return partitions / partitions.sum(dim=1, keepdim=True).clamp_min(1.0)


class StaticDynamicGraphConvolution(nn.Module):
    """结合固定骨架邻接与逐帧数据驱动邻接的图卷积。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        use_static_graph: bool = True,
        use_dynamic_graph: bool = True,
        use_learnable_bias: bool = True,
        use_fixed_prior: bool = True,
    ):
        super().__init__()
        if adjacency.ndim not in (2, 3) or adjacency.shape[-2] != adjacency.shape[-1]:
            raise ValueError("adjacency 必须是 [V,V] 或 [K,V,V]。")
        if not torch.is_floating_point(adjacency):
            raise ValueError("adjacency 必须是浮点张量。")
        if not isinstance(use_static_graph, bool) or not isinstance(use_dynamic_graph, bool):
            raise TypeError("use_static_graph 和 use_dynamic_graph 必须是 bool。")
        if not use_static_graph and not use_dynamic_graph:
            raise ValueError("固定图与动态图至少必须启用一种。")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_points = int(adjacency.shape[-1])
        self.use_static_graph = use_static_graph
        self.use_dynamic_graph = use_dynamic_graph
        self.feature_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.partition_projections = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            for _ in range((adjacency.shape[0] - 1) if use_static_graph and adjacency.ndim == 3 else 0)
        )
        self.use_fixed_prior = use_fixed_prior
        if use_static_graph and use_learnable_bias:
            self.adjacency_bias = nn.Parameter(torch.zeros_like(adjacency))
        else:
            self.register_parameter("adjacency_bias", None)
        if use_dynamic_graph:
            self.query_channels = max(1, in_channels // 4)
            self.query_projection = nn.Conv2d(in_channels, self.query_channels, kernel_size=1)
            self.key_projection = nn.Conv2d(in_channels, self.query_channels, kernel_size=1)
            self.scale = math.sqrt(self.query_channels)
        else:
            self.query_channels = 0
            self.query_projection = None
            self.key_projection = None
            self.scale = None
        self.output_normalization = nn.BatchNorm2d(out_channels)
        self.register_buffer(
            "adjacency", adjacency.detach().clone().float(), persistent=False
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, _, frame_count, point_count = features.shape
        if point_count != self.num_points:
            raise ValueError(
                f"图卷积收到 {point_count} 个关节，但邻接矩阵配置为 {self.num_points} 个。"
            )

        projected = self.feature_projection(features)
        output = torch.zeros_like(projected)
        if self.use_static_graph:
            static_adjacency = self.adjacency
            if not self.use_fixed_prior:
                static_adjacency = torch.zeros_like(static_adjacency)
            if self.adjacency_bias is not None:
                static_adjacency = static_adjacency + self.adjacency_bias
            if static_adjacency.ndim == 2:
                output = output + torch.matmul(projected, static_adjacency)
            else:
                output = output + torch.matmul(projected, static_adjacency[0])
                for index, projection in enumerate(self.partition_projections, 1):
                    output = output + torch.matmul(projection(features), static_adjacency[index])
        if self.use_dynamic_graph:
            query = self.query_projection(features).permute(0, 2, 3, 1)
            query = query.reshape(batch_size * frame_count, point_count, self.query_channels)
            key = self.key_projection(features).permute(0, 2, 1, 3)
            key = key.reshape(batch_size * frame_count, self.query_channels, point_count)
            dynamic_adjacency = torch.softmax(torch.matmul(query, key) / self.scale, dim=-1)
            dynamic_adjacency = dynamic_adjacency.reshape(
                batch_size, frame_count, point_count, point_count
            )
            output = output + torch.matmul(
                projected.permute(0, 2, 1, 3), dynamic_adjacency
            ).permute(0, 2, 1, 3)
        return self.output_normalization(output)


class TemporalBasisConv(nn.Module):
    """时序卷积及可选的标准、傅里叶基或三次 B 样条通道映射。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        temporal_basis: str,
        fourier_degree: int,
        spline_grid_size: int,
    ):
        super().__init__()
        in_channels = _validate_positive_integer("in_channels", in_channels)
        out_channels = _validate_positive_integer("out_channels", out_channels)
        stride = _validate_positive_integer("stride", stride)
        spline_grid_size = _validate_positive_integer("spline_grid_size", spline_grid_size)
        if not isinstance(temporal_basis, str) or temporal_basis not in _TEMPORAL_BASES:
            raise ValueError(
                f"temporal_basis 必须是 {sorted(_TEMPORAL_BASES)} 之一。"
            )
        if (
            isinstance(fourier_degree, bool)
            or not isinstance(fourier_degree, Integral)
            or int(fourier_degree) not in (1, 2, 3)
        ):
            raise ValueError("fourier_degree 必须是 1、2 或 3。")
        fourier_degree = int(fourier_degree)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_basis = temporal_basis
        self.fourier_degree = fourier_degree
        self.spline_grid_size = spline_grid_size

        if temporal_basis == "standard":
            self.standard_convolution = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 1),
                stride=(stride, 1),
                padding=(1, 0),
                bias=False,
            )
            self.standard_normalization = nn.BatchNorm2d(out_channels)
            return

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

        # 趋势支路只做线性通道映射；激活顺序由 forward 中的基函数分支决定。
        self.trend_convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.output_normalization = nn.BatchNorm2d(out_channels)

        if temporal_basis == "fourier":
            self.fourier_coefficients = nn.Parameter(
                torch.randn(fourier_degree, 2, in_channels, out_channels)
                / math.sqrt(in_channels)
            )
        else:
            self.spline_degree = 3
            grid_min, grid_max = -1.0, 1.0
            step = (grid_max - grid_min) / spline_grid_size
            knot_count = spline_grid_size + 1 + 2 * self.spline_degree
            knot_start = grid_min - self.spline_degree * step
            knot_end = grid_max + self.spline_degree * step
            self.register_buffer(
                "spline_knots",
                torch.linspace(knot_start, knot_end, knot_count),
                persistent=False,
            )
            self.spline_basis_count = spline_grid_size + self.spline_degree
            self.spline_coefficients = nn.Parameter(
                torch.randn(in_channels, self.spline_basis_count, out_channels)
                / math.sqrt(in_channels)
            )

    def _fourier_projection(self, features: torch.Tensor) -> torch.Tensor:
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

    def _spline_projection(self, features: torch.Tensor) -> torch.Tensor:
        channel_last = features.permute(0, 2, 3, 1)
        boundary_epsilon = 1e-8
        bounded_features = channel_last.clamp(-1.0, 1.0 - 1e-4).unsqueeze(-1)
        knots = self.spline_knots.to(dtype=channel_last.dtype)

        basis = ((bounded_features >= knots[:-1]) & (bounded_features < knots[1:]))
        basis = basis.to(dtype=channel_last.dtype)
        for degree in range(1, self.spline_degree + 1):
            left_numerator = bounded_features - knots[:-degree - 1]
            left_denominator = knots[degree:-1] - knots[:-degree - 1]
            left_term = (left_numerator / (left_denominator + boundary_epsilon)) * basis[..., :-1]

            right_numerator = knots[degree + 1:] - bounded_features
            right_denominator = knots[degree + 1:] - knots[1:-degree]
            right_term = (right_numerator / (right_denominator + boundary_epsilon)) * basis[..., 1:]
            basis = left_term + right_term

        flattened_basis = basis.reshape(-1, self.in_channels * self.spline_basis_count)
        flattened_coefficients = self.spline_coefficients.reshape(
            self.in_channels * self.spline_basis_count, self.out_channels
        )
        output = torch.matmul(flattened_basis, flattened_coefficients)
        output = output.reshape(channel_last.shape[:-1] + (self.out_channels,))
        return output.permute(0, 3, 1, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.temporal_basis == "standard":
            output = self.standard_convolution(features)
            output = self.standard_normalization(output)
            return F.relu(output, inplace=False)

        basis_features = self.depthwise_normalization(self.temporal_depthwise(features))
        if self.temporal_basis == "fourier":
            basis_features = F.relu(basis_features, inplace=False)
            nonlinear = self._fourier_projection(basis_features)
        else:
            nonlinear = self._spline_projection(basis_features)
        trend = self.trend_convolution(basis_features)
        output = trend + nonlinear
        if self.temporal_basis == "bspline":
            output = F.relu(output, inplace=False)
        return self.output_normalization(output)


class StudentGraphTemporalBlock(nn.Module):
    """由图卷积、时序映射和残差支路组成的学生阶段模块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        stride: int,
        temporal_basis: str,
        fourier_degree: int,
        spline_grid_size: int,
        use_static_graph: bool,
        use_dynamic_graph: bool,
        use_learnable_bias: bool = True,
        use_fixed_prior: bool = True,
    ):
        super().__init__()
        self.graph_convolution = StaticDynamicGraphConvolution(
            in_channels,
            out_channels,
            adjacency,
            use_static_graph=use_static_graph,
            use_dynamic_graph=use_dynamic_graph,
            use_learnable_bias=use_learnable_bias,
            use_fixed_prior=use_fixed_prior,
        )
        self.temporal_convolution = TemporalBasisConv(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=stride,
            temporal_basis=temporal_basis,
            fourier_degree=fourier_degree,
            spline_grid_size=spline_grid_size,
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
                    bias=True,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.residual(features)
        output = self.temporal_convolution(self.graph_convolution(features))
        return output + residual


class ChannelwiseModalityFusion(nn.Module):
    """以全局通道门控融合关节表示和骨向量表示。"""

    def __init__(self, channels: int):
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
        if joint_features.shape != bone_features.shape:
            raise ValueError("融合前的两路特征形状必须相同。")
        weights = self.gate(torch.cat((joint_features, bone_features), dim=1))
        joint_weight, bone_weight = weights.chunk(2, dim=1)
        joint_weight = joint_weight[:, :, None, None]
        bone_weight = bone_weight[:, :, None, None]
        return joint_features * joint_weight + bone_features * bone_weight


class CosineClassifierHead(nn.Module):
    """对归一化特征和类别权重进行余弦分类。"""

    def __init__(self, in_channels: int, num_classes: int, dropout: float):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.weight = nn.Parameter(torch.empty(num_classes, in_channels))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.dropout(self.pool(features).flatten(1))
        normalized_features = F.normalize(pooled, p=2, dim=1)
        normalized_weights = F.normalize(self.weight, p=2, dim=1)
        scale = 1.0 + F.softplus(self.log_scale)
        return scale * (normalized_features @ normalized_weights.t())


class LinearClassifierHead(nn.Module):
    """用于分类头消融的全局池化线性分类器。"""

    def __init__(self, in_channels: int, num_classes: int, dropout: float):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.weight = nn.Parameter(torch.empty(num_classes, in_channels))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.dropout(self.pool(features).flatten(1))
        return F.linear(pooled, self.weight, self.bias)


class MyoFormerStudent(nn.Module):
    """双流轻量学生网络；默认只输出 logits，蒸馏时可选返回六个阶段特征。

    ``joint_coordinates`` 与 ``bone_vectors`` 的输入形状均为 ``[B,C,T,V]``。
    两路分别经过共享拓扑定义的图卷积与时序模块，再进行通道门控融合。
    """

    feature_keys = STUDENT_FEATURE_KEYS

    def __init__(
        self,
        num_classes: int = 33,
        num_points: int = 23,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128),
        dropout: float = 0.5,
        temporal_basis: str = "fourier",
        fourier_degree: int = 1,
        spline_grid_size: int = 3,
        use_static_graph: bool = True,
        use_dynamic_graph: bool = True,
        classifier_head: str = "cosine",
        skeleton_edges: Optional[Sequence[Tuple[int, int]]] = None,
        use_learnable_bias: bool = True,
        use_fixed_prior: bool = True,
    ):
        super().__init__()
        self.num_classes = _validate_positive_integer("num_classes", num_classes)
        if self.num_classes < 2:
            raise ValueError("num_classes 必须至少为 2。")
        self.num_points = _validate_positive_integer("num_points", num_points)
        self.in_channels = _validate_positive_integer("in_channels", in_channels)

        if isinstance(channels, (str, bytes)) or not isinstance(channels, Sequence) or len(channels) != 3:
            raise ValueError("channels 必须包含三个正整数阶段宽度。")
        self.channels = tuple(
            _validate_positive_integer(f"channels[{index}]", value)
            for index, value in enumerate(channels)
        )

        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise ValueError("dropout 必须是 [0,1) 范围内的实数。")
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout 必须是 [0,1) 范围内的有限实数。")
        self.dropout_rate = float(dropout)

        if not isinstance(temporal_basis, str) or temporal_basis not in _TEMPORAL_BASES:
            raise ValueError(
                f"temporal_basis 必须是 {sorted(_TEMPORAL_BASES)} 之一。"
            )
        if (
            isinstance(fourier_degree, bool)
            or not isinstance(fourier_degree, Integral)
            or int(fourier_degree) not in (1, 2, 3)
        ):
            raise ValueError("fourier_degree 必须是 1、2 或 3。")
        self.fourier_degree = int(fourier_degree)
        self.spline_grid_size = _validate_positive_integer(
            "spline_grid_size", spline_grid_size
        )
        self.temporal_basis = temporal_basis
        if not isinstance(use_static_graph, bool) or not isinstance(use_dynamic_graph, bool):
            raise TypeError("use_static_graph 和 use_dynamic_graph 必须是 bool。")
        if not use_static_graph and not use_dynamic_graph:
            raise ValueError("固定图与动态图至少必须启用一种。")
        self.use_static_graph = use_static_graph
        self.use_dynamic_graph = use_dynamic_graph
        if classifier_head not in ("cosine", "linear"):
            raise ValueError("classifier_head 必须是 'cosine' 或 'linear'。")
        self.classifier_head = classifier_head

        adjacency = build_physical_adjacency(self.num_points, skeleton_edges)
        self.register_buffer("physical_adjacency", adjacency, persistent=False)
        adjacency = build_spatial_partitions(self.num_points, skeleton_edges)

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
                    temporal_basis=temporal_basis,
                    fourier_degree=self.fourier_degree,
                    spline_grid_size=self.spline_grid_size,
                    use_static_graph=self.use_static_graph,
                    use_dynamic_graph=self.use_dynamic_graph,
                    use_learnable_bias=use_learnable_bias,
                    use_fixed_prior=use_fixed_prior,
                                )
            )
            self.bone_vector_stream.append(
                StudentGraphTemporalBlock(
                    in_channels=in_stage_channels,
                    out_channels=out_stage_channels,
                    adjacency=adjacency,
                    stride=stride,
                    temporal_basis=temporal_basis,
                    fourier_degree=self.fourier_degree,
                    spline_grid_size=self.spline_grid_size,
                    use_static_graph=self.use_static_graph,
                    use_dynamic_graph=self.use_dynamic_graph,
                    use_learnable_bias=use_learnable_bias,
                    use_fixed_prior=use_fixed_prior,
                                )
            )

        self.fusion = ChannelwiseModalityFusion(self.channels[-1])
        classifier_type = CosineClassifierHead if classifier_head == "cosine" else LinearClassifierHead
        self.classifier = classifier_type(self.channels[-1], self.num_classes, self.dropout_rate)

    def count_params(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _validate_inputs(
        self,
        joint_coordinates: torch.Tensor,
        bone_vectors: torch.Tensor,
    ) -> None:
        for name, tensor in (
            ("joint_coordinates", joint_coordinates),
            ("bone_vectors", bone_vectors),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} 必须是 torch.Tensor。")
            if tensor.ndim != 4:
                raise ValueError(f"{name} 应为 [B,C,T,V]，实际形状为 {tuple(tensor.shape)}。")
            if tensor.shape[0] <= 0 or tensor.shape[2] <= 0:
                raise ValueError(f"{name} 的批大小 B 和时间长度 T 必须大于 0。")
            if tensor.shape[1] != self.in_channels:
                raise ValueError(
                    f"{name} 的通道数为 {tensor.shape[1]}，模型配置为 {self.in_channels}。"
                )
            if tensor.shape[3] != self.num_points:
                raise ValueError(
                    f"{name} 的关节数为 {tensor.shape[3]}，模型配置为 {self.num_points}。"
                )
            if not torch.is_floating_point(tensor):
                raise TypeError(f"{name} 必须使用浮点数据类型。")

        if joint_coordinates.shape[0] != bone_vectors.shape[0]:
            raise ValueError("两路输入的批大小必须相同。")
        if joint_coordinates.shape[2] != bone_vectors.shape[2]:
            raise ValueError("两路输入的时间长度必须相同。")
        if joint_coordinates.device != bone_vectors.device:
            raise ValueError("两路输入必须位于同一设备。")
        if joint_coordinates.dtype != bone_vectors.dtype:
            raise ValueError("两路输入必须使用相同的数据类型。")

        model_parameter = self.classifier.weight
        if joint_coordinates.device != model_parameter.device:
            raise ValueError(
                f"输入位于 {joint_coordinates.device}，模型位于 {model_parameter.device}。"
            )
        if joint_coordinates.dtype != model_parameter.dtype:
            raise TypeError(
                f"输入数据类型为 {joint_coordinates.dtype}，模型参数类型为 {model_parameter.dtype}。"
            )

    def forward(
        self,
        joint_coordinates: torch.Tensor,
        bone_vectors: torch.Tensor,
        return_features: bool = False,
    ):
        if not isinstance(return_features, bool):
            raise TypeError("return_features 必须是 bool。")
        self._validate_inputs(joint_coordinates, bone_vectors)

        joint_features = joint_coordinates
        bone_features = bone_vectors
        stage_features = {}
        for stage_index, (joint_block, bone_block) in enumerate(
            zip(self.joint_stream, self.bone_vector_stream), start=1
        ):
            joint_features = joint_block(joint_features)
            bone_features = bone_block(bone_features)
            if return_features:
                stage_features[f"joint_stage{stage_index}"] = joint_features
                stage_features[f"bone_vector_stage{stage_index}"] = bone_features

        fused_features = self.fusion(joint_features, bone_features)
        logits = self.classifier(fused_features)
        if return_features:
            ordered_features = {
                name: stage_features[name] for name in STUDENT_FEATURE_KEYS
            }
            return logits, ordered_features
        return logits
