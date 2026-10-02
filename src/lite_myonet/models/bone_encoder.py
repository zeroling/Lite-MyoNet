from typing import Optional

import torch
import torch.nn as nn

from .attention.scale_aware_temporal import ScaleAwareTemporalAttention


class MultiScaleBoneTemporalBlock(nn.Module):
    """Fuse parallel temporal receptive fields for the bone-vector stream."""
    def __init__(self, in_channels, out_channels, dropout=0.2, time_stride=1,
                 time_kernel=3, num_heads=8, use_attention=False,
                 attention_config: Optional[dict] = None):
        """Bone-vector temporal encoder block."""
        super().__init__()
        if out_channels % 8 != 0:
            raise ValueError("out_channels must be divisible by 8 to preserve the four-branch channel allocation.")
        if time_stride <= 0:
            raise ValueError("time_stride must be positive.")
        if time_kernel <= 0 or time_kernel % 2 == 0:
            raise ValueError("time_kernel must be a positive odd integer.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.time_stride = time_stride
        self.time_kernel = time_kernel
        self.use_attention = use_attention
        self.attention_config = dict(attention_config or {})
        self.attention_config.setdefault('num_heads', num_heads)

        self.spatial_kernel_sizes = [3, 5, 7, 11]
        self.branch_channels = [out_channels // 8,
                                out_channels // 8,
                                out_channels // 4,
                                out_channels // 2]
        self.total_branch_channels = sum(self.branch_channels)

        self.spatial_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, ch, kernel_size=(1, k), padding=(0, k // 2)),
                nn.BatchNorm2d(ch),
                nn.LeakyReLU(0.1, inplace=True)
            )
            for k, ch in zip(self.spatial_kernel_sizes, self.branch_channels)
        ])

        if self.use_attention:
            self.temporal_attn = ScaleAwareTemporalAttention(
                branch_channels=self.branch_channels,
                dropout_rate=dropout,
                **self.attention_config
            )


        if in_channels == out_channels and time_stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(time_stride, 1)),
                nn.BatchNorm2d(out_channels)
            )


        if time_stride > 1:
            self.time_downsample = nn.Conv2d(
                in_channels=self.total_branch_channels,
                out_channels=self.total_branch_channels,
                kernel_size=(time_kernel, 1),
                stride=(time_stride, 1),
                padding=(time_kernel // 2, 0),
                groups=self.total_branch_channels,
                bias=False
            )
        else:
            self.time_downsample = nn.Identity()

        self.channel_adjust = nn.Sequential(
            nn.Conv2d(self.total_branch_channels + out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Fuse multi-scale bone convolutions, temporal attention, and a residual branch."""
        spatial_outputs = [branch(x) for branch in self.spatial_branches]
        x_spatial = torch.cat(spatial_outputs, dim=1)
        if self.use_attention:
            x_temporal = self.temporal_attn(x_spatial)
        else:
            x_temporal = x_spatial


        x_temporal = self.time_downsample(x_temporal)

        residual = self.residual(x)
        if residual.shape[2:] != x_temporal.shape[2:]:
            raise RuntimeError(
                "Residual and main-branch spatial-temporal dimensions differ: "
                f"{tuple(residual.shape[2:])} versus {tuple(x_temporal.shape[2:])}."
            )

        x_out = torch.cat([x_temporal, residual], dim=1)
        x_out = self.channel_adjust(x_out)
        x_out = self.dropout(x_out)
        return x_out
