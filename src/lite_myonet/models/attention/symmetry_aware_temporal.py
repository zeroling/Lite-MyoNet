import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_position_encoding(T: int, C: int, device: torch.device) -> torch.Tensor:
    """标准 sin/cos 绝对位置编码 [T, C]；若 C 为奇数则末尾补零。"""
    half = C // 2
    pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, half, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / max(half, 1)))
    pe = torch.cat([torch.sin(pos * div), torch.cos(pos * div)], dim=1)  # [T, 2*half]
    if pe.shape[1] < C:
        pe = F.pad(pe, (0, C - pe.shape[1]))
    return pe


def build_contra_index(symmetry_pairs: Optional[List[Tuple[int, int]]],
                       V: int, device: torch.device) -> torch.Tensor:
    """未出现的关节映射到自身。"""
    idx = torch.arange(V, device=device)
    paired_indices = set()
    if symmetry_pairs is not None:
        for a, b in symmetry_pairs:
            if not (0 <= a < V and 0 <= b < V):
                raise ValueError(f"对称关节索引越界：({a}, {b})，关节数为 {V}。")
            if a == b:
                raise ValueError(f"对称关节不能与自身配对：({a}, {b})。")
            if a in paired_indices or b in paired_indices:
                raise ValueError(f"对称关节配对存在重复或冲突：({a}, {b})。")
            paired_indices.update((a, b))
            idx[a] = b
            idx[b] = a
    return idx


class SymmetryAwareTemporalAttention(nn.Module):
    """
    在每个关节的时间序列上计算自注意力和可选的对侧交叉注意力。

    输入和输出均为 ``[B, C, T, V]``。对侧分支的查询来自当前关节，
    键和值来自 ``symmetry_pairs`` 指定的对侧关节；未列出的关节映射到自身。
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        use_contra: bool = True,
        add_pos_to_v: bool = False,
        symmetry_pairs: Optional[List[Tuple[int, int]]] = None,
        pos_gain_init: float = 1.0,
    ):
        super().__init__()
        if num_heads <= 0 or in_channels % num_heads != 0:
            raise ValueError("num_heads 必须大于 0，且 in_channels 必须能被它整除。")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate 必须位于 [0, 1) 区间。")
        self.C = in_channels
        self.H = num_heads
        self.D = in_channels // num_heads

        self.use_contra = bool(use_contra)
        self.add_pos_to_v = bool(add_pos_to_v)
        self.symmetry_pairs = symmetry_pairs

        self.norm = nn.LayerNorm(self.C)
        self.q_proj = nn.Linear(self.C, self.C)
        self.k_proj = nn.Linear(self.C, self.C)
        self.v_proj = nn.Linear(self.C, self.C)
        self.out_proj = nn.Linear(self.C, self.C)
        self.dropout = nn.Dropout(dropout_rate)

        # 非负门控控制绝对位置编码的强度。
        self.pos_gain = nn.Parameter(torch.tensor(float(pos_gain_init), dtype=torch.float32))

        # 可学习门控制显式左右配对关节的对侧交叉注意力贡献。
        self.contra_gate_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @staticmethod
    def _reshape_to_heads(x: torch.Tensor, H: int) -> torch.Tensor:
        """[B*V,T,C] -> [B*V,H,T,D]"""
        Bv, T, C = x.shape
        D = C // H
        return x.view(Bv, T, H, D).permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _scaled_dot(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """q,k: [B*V,H,T,D] -> logits [B*V,H,T,T]"""
        D = q.shape[-1]
        return torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(max(D, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"输入应为 [B,C,T,V]，实际形状为 {tuple(x.shape)}。")
        B, C, T, V = x.shape
        if C != self.C:
            raise ValueError(f"通道数不匹配：输入为 {C}，模型配置为 {self.C}。")
        device = x.device

        # [B,V,T,C] -> [B*V,T,C]
        xv = x.permute(0, 3, 2, 1).contiguous()          # [B,V,T,C]
        x_bvtc = xv.reshape(B * V, T, C)                 # [B*V,T,C]
        x_norm = self.norm(x_bvtc)
        # 绝对位置编码（加到 Q、K；V 可选）
        pe = sinusoidal_position_encoding(T, C, device).to(x_norm.dtype).unsqueeze(0)
        g = self.pos_gain.clamp_min(0.0).view(1, 1, 1)
        x_q = x_norm + g * pe
        x_k = x_norm + g * pe
        x_v = x_norm if not self.add_pos_to_v else (x_norm + g * pe)

        # Q/K/V + 多头
        q = self._reshape_to_heads(self.q_proj(x_q), self.H)  # [B*V,H,T,D]
        k = self._reshape_to_heads(self.k_proj(x_k), self.H)
        v = self._reshape_to_heads(self.v_proj(x_v), self.H)

        # 自注意力
        logits = self._scaled_dot(q, k)                # [B*V,H,T,T]
        logits = logits - logits.amax(dim=-1, keepdim=True)
        attn_self = F.softmax(logits, dim=-1)
        out_self = torch.matmul(self.dropout(attn_self), v)  # [B*V,H,T,D]

        if self.use_contra:
            # 取对侧关节的 K/V
            contra_index = build_contra_index(self.symmetry_pairs, V, device)
            x_norm_v = x_norm.view(B, V, T, C)
            x_kv = x_k.view(B, V, T, C)

            x_k_contra = x_kv[:, contra_index, :, :].reshape(B * V, T, C)
            x_v_contra = (x_norm_v[:, contra_index, :, :].reshape(B * V, T, C)
                          if not self.add_pos_to_v else
                          x_k_contra)

            k_c = self._reshape_to_heads(self.k_proj(x_k_contra), self.H)
            v_c = self._reshape_to_heads(self.v_proj(x_v_contra), self.H)

            # 查询仍来自当前关节，可直接复用自注意力分支的 q。
            logits_c = self._scaled_dot(q, k_c)
            logits_c = logits_c - logits_c.amax(dim=-1, keepdim=True)
            attn_contra = F.softmax(logits_c, dim=-1)
            out_contra = torch.matmul(self.dropout(attn_contra), v_c)

            # 将门控后的对侧信息叠加到关节自身的时间信息。
            gate = torch.sigmoid(self.contra_gate_logit).view(1, 1, 1, 1)
            out = out_self + gate * out_contra
        else:
            out = out_self

        # 还原形状
        out = out.transpose(1, 2).contiguous().view(B * V, T, C)  # [B*V,T,C]
        out = self.out_proj(out)                                  # [B*V,T,C]
        y = out.view(B, V, T, C).permute(0, 3, 2, 1).contiguous() # [B,C,T,V]

        return y
