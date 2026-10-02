import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_position_encoding(T: int, C: int, device: torch.device) -> torch.Tensor:
    """Build time-dependent sine and cosine positional features."""
    half = C // 2
    pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, half, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / max(half, 1)))
    pe = torch.cat([torch.sin(pos * div), torch.cos(pos * div)], dim=1)  # [T, 2*half]
    if pe.shape[1] < C:
        pe = F.pad(pe, (0, C - pe.shape[1]))
    return pe


class ScaleAwareTemporalAttention(nn.Module):
    """Attend over time with heads allocated across multi-scale feature branches."""

    def __init__(
        self,
        branch_channels: List[int],
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        add_pos_to_v: bool = False,
    ):
        """Initialize multi-scale temporal attention and its positional projections."""
        super().__init__()
        self.branch_channels = list(map(int, branch_channels))
        if not self.branch_channels or any(channels <= 0 for channels in self.branch_channels):
            raise ValueError("branch_channels must contain positive integers.")
        self.in_channels = int(sum(self.branch_channels))
        self.num_heads = int(num_heads)
        if self.num_heads < len(self.branch_channels):
            raise ValueError("num_heads must be at least the number of scale branches.")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1).")

        self.heads_per_branch = self._allocate_heads(self.branch_channels, self.num_heads)
        if not all(c >= h for c, h in zip(self.branch_channels, self.heads_per_branch)):
            raise ValueError("Each branch needs at least one channel per assigned attention head.")

        self.branch_head_sizes = [self._even_chunks(c, h)
                                  for c, h in zip(self.branch_channels, self.heads_per_branch)]

        self.add_pos_to_v = bool(add_pos_to_v)

        self.layernorm = nn.LayerNorm(self.in_channels)
        self.W_q = nn.Linear(self.in_channels, self.in_channels)
        self.W_k = nn.Linear(self.in_channels, self.in_channels)
        self.W_v = nn.Linear(self.in_channels, self.in_channels)
        self.W_o = nn.Linear(self.in_channels, self.in_channels)
        self.dropout = nn.Dropout(dropout_rate)


        self.pos_gain = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    @staticmethod
    def _even_chunks(total: int, parts: int) -> List[int]:
        """Divide channels into nearly equal positive head dimensions."""
        base, rem = total // parts, total % parts
        return [base + 1 if i < rem else base for i in range(parts)]

    @staticmethod
    def _allocate_heads(c_list: List[int], H: int) -> List[int]:
        """Allocate attention heads proportionally across temporal-scale branches."""
        S = float(sum(c_list))
        raw = [H * (c / S) for c in c_list]
        floors = [max(1, int(math.floor(x))) for x in raw]
        while sum(floors) > H:
            i = max(range(len(floors)), key=lambda k: floors[k])
            if floors[i] > 1: floors[i] -= 1
            else: break
        while sum(floors) < H:
            frac = [(r - math.floor(r)) for r in raw]
            i = max(range(len(frac)), key=lambda k: (frac[k], -floors[k]))
            floors[i] += 1
        return floors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply branch-specific temporal attention and restore NCTV layout."""
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,T,V] input, got {tuple(x.shape)}.")
        B, C, T, V = x.shape
        if C != self.in_channels:
            raise ValueError(f"Channel count differs: input has {C}, expected {self.in_channels}.")
        device = x.device

        # [B,V,T,C] -> [B*V,T,C]
        x_bt_c = x.permute(0, 3, 2, 1).contiguous().view(B * V, T, C)


        pe = sinusoidal_position_encoding(T, C, device).to(x_bt_c.dtype).unsqueeze(0)
        g = self.pos_gain.clamp_min(0.0).view(1, 1, 1)


        x_norm = self.layernorm(x_bt_c)
        x_qk = x_norm + g * pe
        x_v  = (x_norm + g * pe) if self.add_pos_to_v else x_norm


        q = self.W_q(x_qk)
        k = self.W_k(x_qk)
        v = self.W_v(x_v)


        q_splits = torch.split(q, self.branch_channels, dim=-1)
        k_splits = torch.split(k, self.branch_channels, dim=-1)
        v_splits = torch.split(v, self.branch_channels, dim=-1)

        head_outputs = []
        for b, (qb, kb, vb) in enumerate(zip(q_splits, k_splits, v_splits)):
            sizes = self.branch_head_sizes[b]
            q_heads = list(torch.split(qb, sizes, dim=-1))
            k_heads = list(torch.split(kb, sizes, dim=-1))
            v_heads = list(torch.split(vb, sizes, dim=-1))

            for qh, kh, vh in zip(q_heads, k_heads, v_heads):
                d_h = qh.size(-1)

                scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(max(d_h, 1))
                scores = scores - scores.amax(dim=-1, keepdim=True)
                attn = torch.softmax(scores, dim=-1)                 # [B*V,T,T]
                head_outputs.append(torch.matmul(self.dropout(attn), vh))


        out = torch.cat(head_outputs, dim=-1)  # [B*V,T,C]
        out = self.W_o(out)
        out = out + x_bt_c

        y = out.view(B, V, T, C).permute(0, 3, 2, 1).contiguous()
        return y
