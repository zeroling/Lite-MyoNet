from typing import Optional, List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_connection_mask(num_points, bones, *, add_diag=False,
                           roi_lower=(), roi_gain=1.0):
    """Build a symmetric mask for directly connected skeleton nodes."""
    V = int(num_points)
    M = torch.zeros((V, V), dtype=torch.float32)
    if bones:
        i_idx, j_idx = zip(*bones)
        M[list(i_idx), list(j_idx)] = 1.0
        M[list(j_idx), list(i_idx)] = 1.0
    if add_diag:
        M.fill_diagonal_(1.0)

    if roi_lower and roi_gain != 1.0:
        roi = torch.ones(V, dtype=torch.float32)
        roi[list(roi_lower)] = float(roi_gain)
        M = M * torch.outer(roi, roi)

    return M.unsqueeze(0)


def _build_spatial_distance_mask(num_points, bones, *,
                                 mode="gauss2",
                                 gauss_sigma=1.0,
                                 cutoff=4):
    """Build a graph-distance bias mask for non-neighboring skeleton nodes."""
    V = int(num_points)

    D = torch.full((V, V), float("inf"))
    D.fill_diagonal_(0.0)
    if bones:
        i_idx, j_idx = zip(*bones)
        D[list(i_idx), list(j_idx)] = 1.0
        D[list(j_idx), list(i_idx)] = 1.0
    for k in range(V):
        D = torch.minimum(D, D[:, k].unsqueeze(1) + D[k, :].unsqueeze(0))

    M = torch.zeros((V, V), dtype=torch.float32)
    finite = torch.isfinite(D)

    if mode == "two_hop":
        M[D == 2] = 1.0
        M[D == 3] = 0.5
    elif mode == "inv_dist":
        mask = finite & (D >= 2.0)
        M[mask] = 1.0 / D[mask]
    elif mode == "gauss2":
        mask = finite & (D >= 2.0)
        M[mask] = torch.exp(-((D[mask] - 2.0) ** 2) / (2 * (gauss_sigma ** 2)))
    else:
        raise ValueError(f"Unknown mode: {mode}")


    M[D <= 1.0] = 0.0

    if cutoff is not None:
        M[D > float(cutoff)] = 0.0

    return M.unsqueeze(0)


class StructureAwareSpatialAttention(nn.Module):
    """Apply spatial attention with connectivity, distance, and velocity-divergence biases."""

    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        num_points=23,
        bones: Optional[List[Tuple[int, int]]] = None,
        num_heads=8,
        dropout_rate=0.2,
        mask_scale=1.0,
        eps=1e-5,

        per_head_roles=("free", "conn", "dist", "cdiv"),

        mask_scale_conn=0.9,
        mask_scale_dist=0.7,
        mask_scale_cdiv=1.0,

        cdiv_win=5,
        cdiv_use_abs=True,
        cdiv_speed_power=1.0,
        cdiv_clip=5.0,
        init_tau=1.0,
        spatial_mode="gauss2",
        spatial_gauss_sigma=1.0,
    ):
        """Initialize spatial projections and graph-aware attention masks."""
        super().__init__()
        if num_points <= 0:
            raise ValueError("num_points must be positive.")
        if num_heads <= 0 or hidden_channels % num_heads != 0:
            raise ValueError("num_heads must be positive and divide hidden_channels.")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1).")
        if cdiv_win <= 0 or cdiv_win % 2 == 0:
            raise ValueError("cdiv_win must be a positive odd integer.")
        if init_tau <= 1e-3:
            raise ValueError("init_tau must be greater than 1e-3.")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_points = num_points
        self.num_heads = num_heads
        self.head_channels = hidden_channels // num_heads
        self.eps = float(eps)


        self.fc1 = nn.Linear(in_channels, hidden_channels // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_channels // 2, hidden_channels)
        self.emb_norm = nn.LayerNorm(hidden_channels, eps=eps)


        self.W_q = nn.Linear(hidden_channels, hidden_channels)
        self.W_k = nn.Linear(hidden_channels, hidden_channels)
        self.W_v = nn.Linear(hidden_channels, hidden_channels)
        self.W_o = nn.Linear(hidden_channels, hidden_channels)

        self.attn_dropout_p = float(dropout_rate)


        self.register_buffer('connection_mask',
                             _build_connection_mask(num_points, bones))                          # [1,V,V]
        self.register_buffer('spatial_base_mask',
                             _build_spatial_distance_mask(num_points, bones,
                                                          mode=spatial_mode,
                                                          gauss_sigma=spatial_gauss_sigma))     # [1,V,V]


        self.connection_weights_raw = nn.Parameter(torch.zeros(num_points, num_points))
        self.spatial_weights_raw    = nn.Parameter(torch.zeros(num_points, num_points))


        self.mask_scale = float(mask_scale)
        self.mask_scale_conn = float(mask_scale_conn)
        self.mask_scale_dist = float(mask_scale_dist)
        self.mask_scale_cdiv = float(mask_scale_cdiv)


        self.output_projection = nn.Linear(hidden_channels, in_channels) if in_channels != hidden_channels else nn.Identity()
        self.gamma = nn.Parameter(torch.tensor(0.1))


        allowed_roles = {"free", "conn", "dist", "cdiv"}
        if not per_head_roles:
            raise ValueError("per_head_roles must not be empty.")
        if not all(role in allowed_roles for role in per_head_roles):
            raise ValueError(f"per_head_roles must contain only {sorted(allowed_roles)}.")
        self.per_head_roles = tuple(per_head_roles)
        self.head_gate = nn.Parameter(torch.zeros(num_heads))
        raw_tau = math.log(math.expm1(float(init_tau) - 1e-3))
        self.log_tau = nn.Parameter(torch.tensor(raw_tau, dtype=torch.float32))


        self.cdiv_win = int(cdiv_win)
        self.cdiv_use_abs = bool(cdiv_use_abs)
        self.cdiv_speed_power = float(cdiv_speed_power)
        self.cdiv_clip = None if cdiv_clip is None else float(cdiv_clip)


        self.register_buffer(
            "_idx_cdiv",
            torch.tensor(
                [
                    head_index
                    for head_index in range(self.num_heads)
                    if self.per_head_roles[head_index % len(self.per_head_roles)] == "cdiv"
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self._has_cdiv = self._idx_cdiv.numel() > 0

        self._init_weights(gain=0.1)

    def _init_weights(self, gain=0.1):
        """Initialize attention projections with controlled Xavier scaling."""
        for m in (self.fc1, self.fc2, self.W_q, self.W_k, self.W_v, self.W_o, self.output_projection):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _build_head_masks_static(self, dtype, device):
        """Build role-specific spatial bias masks for all attention heads."""
        V = self.num_points

        conn_mask = self.connection_mask.to(device=device, dtype=dtype).squeeze(0)  # [V,V]
        dist_mask = (1.0 - conn_mask) * self.spatial_base_mask.to(device=device, dtype=dtype).squeeze(0)


        conn_w = torch.tanh(self.connection_weights_raw).to(device=device, dtype=dtype) * (
                    self.mask_scale * self.mask_scale_conn)
        dist_w = torch.tanh(self.spatial_weights_raw).to(device=device, dtype=dtype) * (
                    self.mask_scale * self.mask_scale_dist)

        zero = torch.zeros((V, V), device=device, dtype=dtype)
        base = {
            "free": zero,
            "conn": conn_mask * conn_w,
            "dist": dist_mask * dist_w,
            "cdiv": zero,
        }

        static_masks = []
        for h in range(self.num_heads):
            role = self.per_head_roles[h % len(self.per_head_roles)]
            g = torch.sigmoid(self.head_gate[h])
            static_masks.append(g * base[role])
        static_masks = torch.stack(static_masks, dim=0)  # [H,V,V]
        return static_masks

    def _build_coord_divergence_mask(self, x, eps: float = 1e-6):
        """Measure pairwise divergence of local joint-velocity windows."""
        B, C, T, V = x.shape
        win = max(1, int(self.cdiv_win))
        half = max(0, win // 2)
        device, dtype = x.device, x.dtype


        vel = x.diff(dim=2, prepend=x[:, :, :1, :])


        vel_tlast = vel.permute(0, 1, 3, 2).contiguous()                  # [B,C,V,T]
        vel_pad = F.pad(vel_tlast, (half, half), mode='constant', value=0.0)
        vel_win = vel_pad.unfold(dimension=3, size=win, step=1)            # [B,C,V,T,win]


        t = torch.arange(T, device=device).unsqueeze(1)                    # [T,1]
        u = (torch.arange(win, device=device) - half).view(1, win)         # [1,win]
        mask_tw = ((t + u) >= 0) & ((t + u) < T)                           # [T,win]
        mask_tw_f = mask_tw.to(dtype)                                      # float


        Fv = vel_win.permute(0, 3, 2, 1, 4).reshape(B, T, V, C * win)      # [B,T,V,C*win]
        Fv = Fv / (Fv.norm(dim=-1, keepdim=True) + eps)
        cos = torch.matmul(Fv, Fv.transpose(-1, -2))                        # [B,T,V,V], in [-1,1]
        if self.cdiv_use_abs:
            cos = cos.abs()
        M = 1.0 - cos                                                      # [B,T,V,V]


        sp_raw = torch.linalg.norm(vel_win, dim=1)
        m = mask_tw_f.view(1, 1, T, win)
        sp_sum = (sp_raw * m).sum(dim=-1)                                   # [B,V,T]
        cnt = mask_tw_f.sum(dim=-1).clamp_min(1).view(1, 1, T)              # [1,1,T]
        sp = sp_sum / cnt                                                   # [B,V,T]
        sp = sp.permute(0, 2, 1)                                            # [B,T,V]
        sp = sp / (sp.mean(dim=-1, keepdim=True) + eps)
        if self.cdiv_speed_power != 1.0:
            sp = torch.clamp(sp, min=0).pow(self.cdiv_speed_power)

        gate = sp.unsqueeze(-1) * sp.unsqueeze(-2)                          # [B,T,V,V]
        M = M * gate


        eye = torch.eye(V, device=device, dtype=dtype).view(1, 1, V, V)
        M = M * (1.0 - eye)
        if self.cdiv_clip is not None and self.cdiv_clip > 0:
            M = torch.clamp(M, 0.0, float(self.cdiv_clip))

        return M.reshape(B * T, V, V)

    def forward(self, x):
        """
        x: [B, C, T, V]
        return: [B, C, T, V]
        """
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,T,V] input, got {tuple(x.shape)}.")
        B, C, T, V = x.size()
        if V != self.num_points:
            raise ValueError(f"Joint count differs: input has {V}, expected {self.num_points}.")

        # [B*T, V, C_emb]
        x_flat = x.permute(0, 2, 3, 1).reshape(B * T, V, C)
        x_emb = self.fc2(self.relu(self.fc1(x_flat)))
        x_emb = self.emb_norm(x_emb)

        # Q/K/V -> [B*T, H, V, d]
        H, d = self.num_heads, self.head_channels
        q = self.W_q(x_emb).view(B*T, V, H, d).transpose(1, 2)
        k = self.W_k(x_emb).view(B*T, V, H, d).transpose(1, 2)
        v = self.W_v(x_emb).view(B*T, V, H, d).transpose(1, 2)


        tau = F.softplus(self.log_tau) + 1e-3
        inv_sqrt_tau = torch.rsqrt(tau)
        q = q * inv_sqrt_tau
        k = k * inv_sqrt_tau


        static_masks = self._build_head_masks_static(dtype=q.dtype, device=q.device)  # [H,V,V]


        if self._has_cdiv:
            cdm = self._build_coord_divergence_mask(x, eps=self.eps).to(dtype=q.dtype, device=q.device)  # [B*T,V,V]
            cdm = (self.mask_scale * self.mask_scale_cdiv) * cdm
        else:
            cdm = None


        bias = static_masks.unsqueeze(0).expand(B*T, -1, -1, -1).contiguous()  # [B*T,H,V,V]
        if cdm is not None:
            idx = self._idx_cdiv.to(bias.device)
            bias[:, idx, :, :] = bias[:, idx, :, :] + cdm.unsqueeze(1)


        bias = bias / tau
        bias = bias.reshape(B*T*H, V, V)


        qh = q.reshape(B*T*H, V, d)
        kh = k.reshape(B*T*H, V, d)
        vh = v.reshape(B*T*H, V, d)

        try:
            attn_out = F.scaled_dot_product_attention(
                qh, kh, vh,
                attn_mask=bias,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=False,
            )  # [B*T*H, V, d]
        except TypeError:

            energy = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(max(d, 1))
            energy = energy + bias
            energy = energy - energy.max(dim=-1, keepdim=True)[0]
            attn = F.softmax(energy, dim=-1)
            attn = F.dropout(attn, p=self.attn_dropout_p, training=self.training)
            attn_out = torch.matmul(attn, vh)

        out = attn_out.reshape(B*T, H, V, d).transpose(1, 2).contiguous().view(B*T, V, self.hidden_channels)
        out = self.W_o(out)  # [B*T, V, hidden_channels]
        out = out.view(B, T, V, self.hidden_channels)
        out = self.output_projection(out).permute(0, 3, 1, 2)  # [B, C, T, V]
        y = self.gamma * out + x
        return y
