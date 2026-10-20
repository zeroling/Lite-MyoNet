import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bone_vector_encoder import MultiScaleBoneTemporalBlock
from . import topology
from .attention.structure_aware_spatial import StructureAwareSpatialAttention
from .attention.symmetry_aware_temporal import SymmetryAwareTemporalAttention
from .attention.memory import attention_forward
from .topology import build_adjacency_matrix


def resolve_teacher_width_config(
    width_ratio=1.0,
    spatial_attention_config: Optional[dict] = None,
    symmetry_attention_config: Optional[dict] = None,
    scale_attention_config: Optional[dict] = None,
):
    """Resolve the shared teacher width and matching attention dimensions."""
    ratio = float(width_ratio)
    if not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
        raise ValueError("width_ratio 必须是 (0, 1] 范围内的有限数值。")

    def scale_dimension(value, name):
        scaled = float(value) * ratio
        rounded = int(round(scaled))
        if rounded <= 0 or not math.isclose(scaled, rounded, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "width_ratio={} 无法将 {}={} 缩放为正整数。".format(ratio, name, value)
            )
        return rounded

    stage_channels = tuple(
        scale_dimension(base, "teacher stage channel")
        for base in (64, 128, 256)
    )
    base_heads = 8
    if any(channels % 8 != 0 for channels in stage_channels):
        raise ValueError("width_ratio 缩放后的教师通道必须能被骨向量多尺度分支的 8 整除。")
    if any(channels % base_heads != 0 for channels in stage_channels):
        raise ValueError("width_ratio 缩放后的教师通道必须能被缩放后的注意力头数整除。")
    if base_heads < 4:
        raise ValueError("width_ratio 缩放后的注意力头数必须至少为 4，以覆盖骨向量四个尺度分支。")

    def scaled_attention_options(config, name):
        options = dict(config or {})
        requested_heads = int(options.get("num_heads", 8))
        options["num_heads"] = requested_heads
        return options

    spatial = scaled_attention_options(spatial_attention_config, "spatial_attention")
    spatial_hidden = int(spatial.get("hidden_channels", 64))
    spatial["hidden_channels"] = scale_dimension(
        spatial_hidden, "spatial_attention.hidden_channels"
    )
    if spatial["hidden_channels"] % spatial["num_heads"] != 0:
        raise ValueError(
            "缩放后的 spatial_attention.hidden_channels 必须能被 num_heads 整除。"
        )

    symmetry = scaled_attention_options(symmetry_attention_config, "symmetry_attention")
    scale = scaled_attention_options(scale_attention_config, "scale_attention")
    if scale["num_heads"] < 4:
        raise ValueError("scale_attention.num_heads 不能少于多尺度骨向量分支数 4。")

    return {
        "width_ratio": ratio,
        "stage_channels": stage_channels,
        "num_heads": base_heads,
        "spatial_attention": spatial,
        "symmetry_attention": symmetry,
        "scale_attention": scale,
    }


class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, A, bias=True):
        super(GraphConvolution, self).__init__()
        self.kernel_size = kernel_size
        # 保留旧实现的卷积偏置与图聚合后的输出偏置。
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=1,
            bias=True,
        )
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # 邻接矩阵随模型迁移设备，但它可由骨架拓扑重新生成，无需写入权重文件。
        # persistent=False 同时保证与旧版权重的键集合兼容。
        self.register_buffer(
            'A', torch.as_tensor(A, dtype=torch.float32).detach().clone(), persistent=False
        )  # [K, V, V]

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A))
        if self.bias is not None:
            return x + self.bias.view(1, -1, 1, 1)
        else:
            return x


class JointTemporalAttentionBlock(nn.Module):
    """
    图卷积与对称性感知时间注意力组成的预激活残差块。

    归一化和激活位于图卷积之前；时间步幅为 2 时执行下采样。
    ``attention_config`` 原样传给时间注意力模块，未知字段会直接报错。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout, A,
                 use_attention=False, use_groupnorm=False, num_groups=32,
                 attention_config: Optional[dict] = None,
                 symmetry_pairs=None):
        super().__init__()
        self.use_attention = use_attention
        self.stride = stride
        self.dropout = dropout
        self.attention_config = dict(attention_config or {})
        if symmetry_pairs is not None:
            self.attention_config.setdefault('symmetry_pairs', symmetry_pairs)

        # 按配置选择批归一化或组归一化。
        if use_groupnorm:
            g1 = min(num_groups, in_channels)
            g2 = min(num_groups, out_channels)
            self.norm1 = nn.GroupNorm(num_groups=g1, num_channels=in_channels)
            self.norm2 = nn.GroupNorm(num_groups=g2, num_channels=out_channels)
            self._use_groupnorm = True
        else:
            self.norm1 = nn.BatchNorm2d(in_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
            self._use_groupnorm = False

        # 预激活图卷积。
        self.gcn_pre = nn.Sequential(
            self.norm1,
            nn.ReLU(inplace=True),
            GraphConvolution(in_channels, out_channels, kernel_size[0], A),
            self.norm2,
            nn.ReLU(inplace=True)
        )

        if self.use_attention:
            self.temporal_attn = SymmetryAwareTemporalAttention(
                out_channels, dropout_rate=dropout, **self.attention_config
            )

        if stride == 2:
            self.temporal_conv = nn.Conv2d(
                out_channels, out_channels,
                kernel_size=(kernel_size[1], 1),
                stride=(stride, 1),
                padding=(kernel_size[1] // 2, 0)
            )
            nn.init.kaiming_normal_(self.temporal_conv.weight, mode='fan_out', nonlinearity='relu')
            if self.temporal_conv.bias is not None:
                nn.init.constant_(self.temporal_conv.bias, 0)
        else:
            self.temporal_conv = nn.Identity()

        # 通道数或时间分辨率变化时对残差分支做投影。
        if stride == 2 or in_channels != out_channels:
            res_layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1))]
            if self._use_groupnorm:
                res_layers.append(nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels))
            else:
                res_layers.append(nn.BatchNorm2d(out_channels))
            self.residual = nn.Sequential(*res_layers)
        else:
            self.residual = nn.Sequential()

    def forward(self, x):
        residual = self.residual(x)

        x = self.gcn_pre(x)

        if self.use_attention:
            x = attention_forward(self.temporal_attn, x)

        x = self.temporal_conv(x)

        # 残差相加后再执行随机失活。
        x = x + residual
        x = F.dropout(x, self.dropout, self.training)
        return x


class JointSpatialAttentionBlock(nn.Module):
    """
    结构感知空间注意力与时间卷积组成的预激活残差块。

    ``attention_config`` 原样传给空间注意力模块，未知字段会直接报错。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout,
                 num_points, use_attention=False, use_groupnorm=False,
                 attention_config: Optional[dict] = None, skeleton_edges=None):
        super().__init__()
        self.use_attention = use_attention
        self.use_groupnorm = use_groupnorm
        self.dropout = dropout
        self.attention_config = dict(attention_config or {})

        if self.use_attention:
            self.spatial_attn = StructureAwareSpatialAttention(
                in_channels,
                num_points=num_points,
                bones=topology.bones if skeleton_edges is None else skeleton_edges,
                dropout_rate=dropout,
                **self.attention_config
            )

        # 时间卷积采用“归一化—激活—卷积”的预激活顺序。
        if use_groupnorm:
            ng1 = min(32, in_channels)
            ng2 = min(32, out_channels)
            self.norm_in = nn.GroupNorm(num_groups=ng1, num_channels=in_channels)
            self.norm_out = nn.GroupNorm(num_groups=ng2, num_channels=out_channels)
        else:
            self.norm_in = nn.BatchNorm2d(in_channels)
            self.norm_out = nn.BatchNorm2d(out_channels)

        self.tcn_conv = nn.Conv2d(in_channels, out_channels, (kernel_size[1], 1), (stride, 1),
                                  padding=((kernel_size[1] - 1) // 2, 0))
        nn.init.kaiming_normal_(self.tcn_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.tcn_conv.bias is not None:
            nn.init.constant_(self.tcn_conv.bias, 0)

        # 通道数或时间分辨率变化时对残差分支做投影。
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Sequential()
        else:
            res_layers = [nn.Conv2d(in_channels, out_channels, 1, (stride, 1))]
            res_layers.append(nn.BatchNorm2d(out_channels) if not use_groupnorm else nn.GroupNorm(num_groups=min(32,out_channels), num_channels=out_channels))
            self.residual = nn.Sequential(*res_layers)

    def forward(self, x):
        residual = self.residual(x)
        if self.use_attention:
            x = attention_forward(self.spatial_attn, x)

        x = self.norm_in(x)
        x = F.relu(x, inplace=True)
        x = self.tcn_conv(x)
        x = self.norm_out(x)

        x = x + residual
        x = F.dropout(x, self.dropout, self.training)
        return x


class SpatioTemporalGraphBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout, A, use_groupnorm=False):
        super().__init__()
        self.use_groupnorm = use_groupnorm
        if use_groupnorm:
            self.gcn_norm = nn.GroupNorm(num_groups=min(32,in_channels), num_channels=in_channels)
            self.tcn_norm = nn.GroupNorm(num_groups=min(32,out_channels), num_channels=out_channels)
        else:
            self.gcn_norm = nn.BatchNorm2d(in_channels)
            self.tcn_norm = nn.BatchNorm2d(out_channels)

        self.gcn = nn.Sequential(
            self.gcn_norm,
            nn.ReLU(inplace=True),
            GraphConvolution(in_channels, out_channels, kernel_size[0], A),
        )
        self.tcn_conv = nn.Conv2d(out_channels, out_channels, (kernel_size[1], 1), (stride, 1),
                                  padding=((kernel_size[1] - 1) // 2, 0))
        nn.init.kaiming_normal_(self.tcn_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.tcn_conv.bias is not None:
            nn.init.constant_(self.tcn_conv.bias, 0)

        # 通道数或时间分辨率变化时对残差分支做投影。
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Sequential()
        else:
            res_layers = [nn.Conv2d(in_channels, out_channels, 1, (stride, 1))]
            res_layers.append(nn.BatchNorm2d(out_channels) if not use_groupnorm else nn.GroupNorm(num_groups=min(32,out_channels), num_channels=out_channels))
            self.residual = nn.Sequential(*res_layers)

        self.dropout = dropout

    def forward(self, x):
        residual = self.residual(x)
        x = self.gcn(x)
        x = self.tcn_conv(x)
        x = F.dropout(x, self.dropout, self.training)
        return x + residual


class CosineClassifierHead(nn.Module):
    """
    余弦分类头：对特征与权重做 L2 归一化，再乘以可学习尺度 s（控制分布“尖/钝”）。
    - s 小 -> 概率更平缓；s 大 -> 概率更尖锐。
    - 支持 LayerNorm / BatchNorm、(MC) Dropout 多样本平均。
    """
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
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.num_mc_samples = int(num_mc_samples) if num_mc_samples is not None else 0
        self.mc_dropout_eval = bool(mc_dropout_eval)
        if self.num_mc_samples < 0:
            raise ValueError("num_mc_samples 不能为负数。")
        if scale_min < 0:
            raise ValueError("scale_min 不能为负数。")
        if scale_init < scale_min:
            raise ValueError("scale_init 不能小于 scale_min。")
        if scale_max is not None and scale_max <= scale_min:
            raise ValueError("scale_max 必须大于 scale_min。")
        if scale_max is not None and scale_init > scale_max:
            raise ValueError("scale_init 不能大于 scale_max。")

        # 分类前的特征归一化。
        norm = (norm or "").lower()
        if norm == "layernorm":
            self.norm = nn.LayerNorm(in_channels)
        elif norm == "batchnorm":
            self.norm = nn.BatchNorm1d(in_channels)
        else:
            self.norm = None

        # 分类权重在前向传播中执行 L2 归一化。
        self.weight = nn.Parameter(torch.empty(num_classes, in_channels))
        nn.init.kaiming_normal_(self.weight, nonlinearity='linear')

        # 复现旧代码的 log(scale_init-scale_min) 初始化约定。
        scale_increment = max(1e-6, float(scale_init) - float(scale_min))
        raw_scale = math.log(scale_increment)
        self.log_scale = nn.Parameter(torch.tensor(raw_scale, dtype=torch.float32))
        self.register_buffer("scale_min", torch.tensor(float(scale_min)))
        self.register_buffer("scale_max", torch.tensor(float(scale_max if scale_max is not None else 0.0)))
        self.has_scale_max = scale_max is not None and float(scale_max) > 0.0

    def _compute_logits(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, C]
        z = F.normalize(z, dim=1)                # 特征归一化
        w = F.normalize(self.weight, dim=1)      # 权重归一化
        s = self.scale_min + F.softplus(self.log_scale)
        if self.has_scale_max:
            s = torch.clamp(s, max=self.scale_max)
        return s * (z @ w.t())

    def _apply_dropout(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(
            x,
            p=self.dropout.p,
            training=self.training or self.mc_dropout_eval,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, V] 或 [B, C, 1, 1]
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

class MyoFormerTeacher(nn.Module):
    """X1/X2/X2 三流教师：一路关节坐标、两路骨向量。"""

    def __init__(self, num_classes=33, num_points=23, in_channels=3, dropout=0.05,
                 use_spatial_attention=True,
                 use_symmetry_attention=True,
                 use_scale_attention=True,
                 head_dropout=0.3,
                 spatial_attention_config: Optional[dict] = None,
                 symmetry_attention_config: Optional[dict] = None,
                 scale_attention_config: Optional[dict] = None,
                 head_config: Optional[dict] = None,
                 use_joint_spatial_stream=True,
                 use_joint_temporal_stream=True,
                 use_bone_vector_stream=True,
                 width_ratio=1.0,
                 skeleton_edges=None,
                 symmetry_pairs=None,
                 root_index=0,
                 input_scheme="X1/X2/X2",
                 ):
        super().__init__()
        if input_scheme not in ("X1/X2/X2", "X1/X1/X2"):
            raise ValueError("教师input_scheme必须为X1/X2/X2或X1/X1/X2。")
        self.input_scheme = input_scheme
        width_config = resolve_teacher_width_config(
            width_ratio,
            spatial_attention_config,
            symmetry_attention_config,
            scale_attention_config,
        )
        stage1_channels, stage2_channels, stage3_channels = width_config["stage_channels"]
        self.width_ratio = width_config["width_ratio"]
        self.num_points = num_points
        self.in_channels = in_channels
        self.bones = list(topology.bones if skeleton_edges is None else skeleton_edges)
        self.symmetry_pairs = tuple(topology.symmetry_pairs if symmetry_pairs is None else symmetry_pairs)
        self.register_buffer(
            'A', build_adjacency_matrix(num_points, self.bones, center=int(root_index)), persistent=False
        )
        self.num_classes = num_classes
        self.use_joint_spatial_stream = bool(use_joint_spatial_stream)
        self.use_joint_temporal_stream = bool(use_joint_temporal_stream)
        self.use_bone_vector_stream = bool(use_bone_vector_stream)
        if not any(
            (
                self.use_joint_spatial_stream,
                self.use_joint_temporal_stream,
                self.use_bone_vector_stream,
            )
        ):
            raise ValueError("至少需要启用一个输入流。")

        self.spatial_attention_config = width_config["spatial_attention"]
        self.symmetry_attention_config = width_config["symmetry_attention"]
        self.scale_attention_config = width_config["scale_attention"]

        if self.use_joint_spatial_stream:
            self.joint_spatial_stream = nn.ModuleList([
                SpatioTemporalGraphBlock(in_channels, stage1_channels, (3, 3), 1, dropout, self.A),
                SpatioTemporalGraphBlock(stage1_channels, stage1_channels, (3, 3), 2, dropout, self.A),
                SpatioTemporalGraphBlock(stage1_channels, stage1_channels, (3, 3), 1, dropout, self.A),
                JointSpatialAttentionBlock(
                    stage1_channels, stage2_channels, (3, 3), 1, dropout, num_points,
                    use_attention=use_spatial_attention,
                    attention_config=self.spatial_attention_config,
                    skeleton_edges=self.bones,
                ),
                JointSpatialAttentionBlock(
                    stage2_channels, stage2_channels, (3, 3), 1, dropout, num_points,
                    use_attention=use_spatial_attention,
                    attention_config=self.spatial_attention_config,
                    skeleton_edges=self.bones,
                ),
                JointSpatialAttentionBlock(
                    stage2_channels, stage2_channels, (3, 3), 1, dropout, num_points,
                    use_attention=use_spatial_attention,
                    attention_config=self.spatial_attention_config,
                    skeleton_edges=self.bones,
                ),
                JointSpatialAttentionBlock(
                    stage2_channels, stage3_channels, (3, 3), 2, dropout, num_points,
                    use_attention=use_spatial_attention,
                    attention_config=self.spatial_attention_config,
                    skeleton_edges=self.bones,
                ),
                JointSpatialAttentionBlock(
                    stage3_channels, stage3_channels, (3, 3), 1, dropout, num_points,
                    use_attention=use_spatial_attention,
                    attention_config=self.spatial_attention_config,
                    skeleton_edges=self.bones,
                ),
                JointSpatialAttentionBlock(
                    stage3_channels, stage3_channels, (3, 3), 2, dropout, num_points,
                    use_attention=use_spatial_attention,
                    attention_config=self.spatial_attention_config,
                    skeleton_edges=self.bones,
                ),
            ])

        if self.use_joint_temporal_stream:
            self.joint_temporal_stream = nn.ModuleList([
                SpatioTemporalGraphBlock(in_channels, stage1_channels, (3, 3), 1, dropout, self.A),
                SpatioTemporalGraphBlock(stage1_channels, stage1_channels, (3, 3), 2, dropout, self.A),
                SpatioTemporalGraphBlock(stage1_channels, stage1_channels, (3, 3), 1, dropout, self.A),
                *[
                    JointTemporalAttentionBlock(
                        input_channels,
                        output_channels,
                        (3, 3),
                        stride,
                        dropout,
                        self.A,
                        use_attention=use_symmetry_attention,
                        attention_config=self.symmetry_attention_config,
                        symmetry_pairs=self.symmetry_pairs,
                    )
                    for input_channels, output_channels, stride in (
                        (stage1_channels, stage2_channels, 1),
                        (stage2_channels, stage2_channels, 1),
                        (stage2_channels, stage2_channels, 1),
                        (stage2_channels, stage3_channels, 2),
                        (stage3_channels, stage3_channels, 1),
                        (stage3_channels, stage3_channels, 2),
                    )
                ],
            ])

        if self.use_bone_vector_stream:
            self.bone_vector_stream = nn.ModuleList([
                MultiScaleBoneTemporalBlock(in_channels, stage1_channels, dropout),
                MultiScaleBoneTemporalBlock(stage1_channels, stage1_channels, dropout, time_stride=2),
                MultiScaleBoneTemporalBlock(
                    stage1_channels, stage1_channels, dropout, use_attention=use_scale_attention,
                    attention_config=self.scale_attention_config,
                ),
                MultiScaleBoneTemporalBlock(
                    stage1_channels, stage2_channels, dropout, use_attention=use_scale_attention,
                    attention_config=self.scale_attention_config,
                ),
                MultiScaleBoneTemporalBlock(
                    stage2_channels, stage2_channels, dropout, use_attention=use_scale_attention,
                    attention_config=self.scale_attention_config,
                ),
                MultiScaleBoneTemporalBlock(
                    stage2_channels, stage2_channels, dropout, use_attention=use_scale_attention,
                    attention_config=self.scale_attention_config,
                ),
                MultiScaleBoneTemporalBlock(
                    stage2_channels, stage3_channels, dropout, time_stride=2, use_attention=use_scale_attention,
                    attention_config=self.scale_attention_config,
                ),
                MultiScaleBoneTemporalBlock(
                    stage3_channels, stage3_channels, dropout, use_attention=use_scale_attention,
                    attention_config=self.scale_attention_config,
                ),
                MultiScaleBoneTemporalBlock(
                    stage3_channels, stage3_channels, dropout, time_stride=2, use_attention=use_scale_attention,
                    attention_config=self.scale_attention_config,
                ),
            ])

        # 三个流共用同一个分类头；消融实验可通过配置切换头类型。
        hcfg = head_config or {}
        head_type = str(hcfg.get('type', 'cosine')).lower()

        if head_type == 'cosine':
            self.classifier = CosineClassifierHead(
                in_channels=stage3_channels, num_classes=num_classes,
                dropout=hcfg.get('dropout', head_dropout),
                scale_init=hcfg.get('scale_init', 10.0),
                scale_min=hcfg.get('scale_min', 1.0),
                scale_max=hcfg.get('scale_max', 20.0),
                norm=hcfg.get('norm', 'layernorm'),
                num_mc_samples=int(hcfg.get('num_mc_samples', 0)),
                mc_dropout_eval=bool(hcfg.get('mc_dropout_eval', False)),
            )
        elif head_type == 'linear_wn':
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(head_dropout),
                nn.utils.weight_norm(nn.Linear(stage3_channels, num_classes))
            )
        elif head_type == 'linear':
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(head_dropout),
                nn.Linear(stage3_channels, num_classes)
            )
        else:
            raise ValueError(
                f"Unsupported classifier head type: {head_type!r}. "
                "Expected one of: cosine, linear_wn, linear."
            )

    @staticmethod
    def _forward_stream(x, layers, feature_indices=()):
        features = []
        capture = set(feature_indices)
        for index, layer in enumerate(layers):
            x = layer(x)
            if index in capture:
                features.append(x)
        return x, features

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _validate_input(self, name, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} 必须是 torch.Tensor。")
        if tensor.ndim != 4:
            raise ValueError(f"{name} 应为 [B,C,T,V]，实际形状为 {tuple(tensor.shape)}。")
        if tensor.shape[1] != self.in_channels:
            raise ValueError(
                f"{name} 的通道数为 {tensor.shape[1]}，模型配置为 {self.in_channels}。"
            )
        if tensor.shape[3] != self.num_points:
            raise ValueError(
                f"{name} 的关节数为 {tensor.shape[3]}，模型配置为 {self.num_points}。"
            )

    def forward(self, joint_coordinates, bone_vectors, return_features=False):
        """
        默认三路输入为X1/X2/X2；input_scheme为X1/X1/X2时，
        第二路时间分支改用joint_coordinates，其余两路及全部模块保持相同。

        return_features=False 用于教师独立训练和推理，只返回 logits。
        return_features=True 用于蒸馏，对每个启用的分支返回 3 个阶段特征；
        三个分支全部启用时共返回 9 个特征。
        """
        streams = []
        teacher_features = []
        feature_indices = (3, 6, 8) if return_features else ()

        uses_joint_coordinates = self.use_joint_spatial_stream or (
            self.use_joint_temporal_stream and self.input_scheme == "X1/X1/X2"
        )
        if uses_joint_coordinates:
            self._validate_input("joint_coordinates", joint_coordinates)
        if self.use_bone_vector_stream or (
            self.use_joint_temporal_stream and self.input_scheme == "X1/X2/X2"
        ):
            self._validate_input("bone_vectors", bone_vectors)
        if uses_joint_coordinates and self.use_bone_vector_stream:
            joint_shape = (
                joint_coordinates.shape[0],
                joint_coordinates.shape[2],
                joint_coordinates.shape[3],
            )
            bone_shape = (
                bone_vectors.shape[0],
                bone_vectors.shape[2],
                bone_vectors.shape[3],
            )
            if joint_shape != bone_shape:
                raise ValueError(
                    "joint_coordinates 与 bone_vectors 的批大小、时间长度和关节数必须一致，"
                    f"实际分别为 {joint_shape} 和 {bone_shape}。"
                )

        if self.use_joint_spatial_stream:
            output, features = self._forward_stream(
                joint_coordinates, self.joint_spatial_stream, feature_indices
            )
            streams.append(output)
            teacher_features.extend(features)

        if self.use_joint_temporal_stream:
            temporal_input = joint_coordinates if self.input_scheme == "X1/X1/X2" else bone_vectors
            output, features = self._forward_stream(
                temporal_input, self.joint_temporal_stream, feature_indices
            )
            streams.append(output)
            teacher_features.extend(features)

        if self.use_bone_vector_stream:
            output, features = self._forward_stream(
                bone_vectors, self.bone_vector_stream, feature_indices
            )
            streams.append(output)
            teacher_features.extend(features)

        if not streams:
            raise RuntimeError("至少需要启用一个流")

        fused = sum(streams)
        class_out = self.classifier(fused)
        if return_features:
            return class_out, teacher_features
        return class_out
