import math
from typing import List, Optional, Tuple

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


def build_contra_index(symmetry_pairs: Optional[List[Tuple[int, int]]],
                       V: int, device: torch.device) -> torch.Tensor:
    """Map each joint to its contralateral partner, leaving unpaired joints unchanged."""
    idx = torch.arange(V, device=device)
    paired_indices = set()
    if symmetry_pairs is not None:
        for a, b in symmetry_pairs:
            if not (0 <= a < V and 0 <= b < V):
                raise ValueError(f"Symmetry pair ({a}, {b}) is outside the {V}-joint skeleton.")
            if a == b:
                raise ValueError(f"A joint cannot be paired with itself: ({a}, {b}).")
            if a in paired_indices or b in paired_indices:
                raise ValueError(f"Duplicate or conflicting symmetry pair: ({a}, {b}).")
            paired_indices.update((a, b))
            idx[a] = b
            idx[b] = a
    return idx


class SymmetryAwareTemporalAttention(nn.Module):
    """Combine within-joint and contralateral-joint temporal attention."""

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
        """Initialize temporal attention with configured left-right joint correspondences."""
        super().__init__()
        if num_heads <= 0 or in_channels % num_heads != 0:
            raise ValueError("num_heads must be positive and divide in_channels.")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1).")
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


        self.pos_gain = nn.Parameter(torch.tensor(float(pos_gain_init), dtype=torch.float32))


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
        """Combine self and contralateral temporal attention in NCTV layout."""
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,T,V] input, got {tuple(x.shape)}.")
        B, C, T, V = x.shape
        if C != self.C:
            raise ValueError(f"Channel count differs: input has {C}, expected {self.C}.")
        device = x.device

        # [B,V,T,C] -> [B*V,T,C]
        xv = x.permute(0, 3, 2, 1).contiguous()          # [B,V,T,C]
        x_bvtc = xv.reshape(B * V, T, C)                 # [B*V,T,C]
        x_norm = self.norm(x_bvtc)

        pe = sinusoidal_position_encoding(T, C, device).to(x_norm.dtype).unsqueeze(0)
        g = self.pos_gain.clamp_min(0.0).view(1, 1, 1)
        x_q = x_norm + g * pe
        x_k = x_norm + g * pe
        x_v = x_norm if not self.add_pos_to_v else (x_norm + g * pe)


        q = self._reshape_to_heads(self.q_proj(x_q), self.H)  # [B*V,H,T,D]
        k = self._reshape_to_heads(self.k_proj(x_k), self.H)
        v = self._reshape_to_heads(self.v_proj(x_v), self.H)


        logits = self._scaled_dot(q, k)                # [B*V,H,T,T]
        logits = logits - logits.amax(dim=-1, keepdim=True)
        attn_self = F.softmax(logits, dim=-1)
        out_self = torch.matmul(self.dropout(attn_self), v)  # [B*V,H,T,D]

        if self.use_contra:

            contra_index = build_contra_index(self.symmetry_pairs, V, device)
            x_norm_v = x_norm.view(B, V, T, C)
            x_kv = x_k.view(B, V, T, C)

            x_k_contra = x_kv[:, contra_index, :, :].reshape(B * V, T, C)
            x_v_contra = (x_norm_v[:, contra_index, :, :].reshape(B * V, T, C)
                          if not self.add_pos_to_v else
                          x_k_contra)

            k_c = self._reshape_to_heads(self.k_proj(x_k_contra), self.H)
            v_c = self._reshape_to_heads(self.v_proj(x_v_contra), self.H)


            logits_c = self._scaled_dot(q, k_c)
            logits_c = logits_c - logits_c.amax(dim=-1, keepdim=True)
            attn_contra = F.softmax(logits_c, dim=-1)
            out_contra = torch.matmul(self.dropout(attn_contra), v_c)


            paired_mask = contra_index.ne(torch.arange(V, device=device))
            out_contra = out_contra.view(B, V, self.H, T, self.D)
            out_contra = out_contra * paired_mask.view(1, V, 1, 1, 1).to(out_contra.dtype)
            out_contra = out_contra.view(B * V, self.H, T, self.D)


            gate = torch.sigmoid(self.contra_gate_logit).view(1, 1, 1, 1)
            out = out_self + gate * out_contra
        else:
            out = out_self


        out = out.transpose(1, 2).contiguous().view(B * V, T, C)  # [B*V,T,C]
        out = self.out_proj(out)                                  # [B*V,T,C]
        y = out.view(B, V, T, C).permute(0, 3, 2, 1).contiguous() # [B,C,T,V]

        return y
