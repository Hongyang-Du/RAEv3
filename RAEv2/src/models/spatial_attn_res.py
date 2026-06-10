"""
SpatialAttnRes: last DINOv3 layer as base, intermediate layers injected via
cross-attention + learnable gate.

For each of 256 spatial positions independently:
  Q = last layer token           (base representation)
  K/V = intermediate layer tokens (layers 0..K-2, NOT last)
  cross_attn = softmax(Q @ K^T / sqrt(d)) @ V
  gate = sigmoid(learnable)      (init 0.5, open & learnable; or randomly open)
  z = last_layer + gate × cross_attn
  z = FFN(z)                     (learn new representation space)
  z = BatchNorm(z)               (compatible with SIGReg)

K=1 (only last layer) → no intermediate layers → z = FFN(last_layer)
"""

import torch
import torch.nn as nn


RAEV2_LAYERS = [11, 13, 15, 17, 19, 21, 23]   # same as RAEv2 MLS K7


class SpatialAttnRes(nn.Module):
    def __init__(self, dim: int = 1024, out_dim: int = 1024,
                 gate_init: float = 0.0, gate_random: bool = False):
        """
        dim         : DINOv3 hidden dim (1024 for ViT-L)
        out_dim     : output latent dim
        gate_init   : initial gate logit. sigmoid(0)=0.5 → open AND at the max-gradient
                      point of the sigmoid, so the gate is immediately learnable.
                      (sigmoid(-5)≈0.007 starts ~closed and saturated → barely moves.)
        gate_random : if True, per-dim logits ~ N(gate_init, 1) — "randomly open",
                      breaks symmetry across the 1024 gates so each dim starts at a
                      different openness. Synced across ranks by DDP's init broadcast.
        """
        super().__init__()
        self.scale  = dim ** -0.5

        # cross-attention projections (per-position, shared across layers)
        self.q_proj = nn.Linear(dim, dim, bias=False)  # from last layer
        self.k_proj = nn.Linear(dim, dim, bias=False)  # from intermediate layers
        # no v_proj — use intermediate layer outputs directly as values

        # gate: one logit per output dim, controls how much intermediate info to inject
        if gate_random:
            self.gate = nn.Parameter(torch.randn(dim) + gate_init)   # randomly open
        else:
            self.gate = nn.Parameter(torch.full((dim,), gate_init))  # open at sigmoid(gate_init)

        # FFN: learns the new representation space on top of fused features
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, out_dim),
        )

        # BatchNorm at output — compatible with SIGReg (cross-batch statistics)
        self.out_norm = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, patches_list: list[torch.Tensor]) -> torch.Tensor:
        """
        patches_list : K tensors each [B, 256, dim],
                       ordered [layer_earliest, ..., layer_last]
        returns      : z [B, 256, out_dim]
        """
        B, N, D = patches_list[0].shape

        last_layer    = patches_list[-1]          # [B, N, D]  ← base
        inter_layers  = patches_list[:-1]         # K-1 tensors, intermediate

        # ── per-position reshape → [B*N, ...]  ───────────────────────────────
        q_in = last_layer.reshape(B * N, 1, D)   # [B*N, 1, D]

        if inter_layers:
            # stack intermediate: [B, K-1, N, D] → [B*N, K-1, D]
            kv_in = (torch.stack(inter_layers, dim=1)  # [B, K-1, N, D]
                     .transpose(1, 2)                   # [B, N, K-1, D]
                     .reshape(B * N, len(inter_layers), D))

            # cross-attention: Q=last_layer, K/V=intermediate
            q = self.q_proj(q_in)                # [B*N, 1, D]
            k = self.k_proj(kv_in)               # [B*N, K-1, D]
            w = torch.softmax(
                (q @ k.transpose(-2, -1)) * self.scale, dim=-1  # [B*N, 1, K-1]
            )
            cross = (w @ kv_in).squeeze(1)       # [B*N, D]  (values = raw inter)
            cross = cross.reshape(B, N, D)       # [B, N, D]

            # gated injection into last layer
            gate = torch.sigmoid(self.gate)      # [D], init ≈ 0
            z = last_layer + gate * cross        # [B, N, D]
        else:
            # K=1: only last layer, no intermediate → skip cross-attn
            z = last_layer                       # [B, N, D]

        # ── FFN: new representation space ────────────────────────────────────
        z = z + self.ffn(self.ffn_norm(z))       # [B, N, out_dim]  (residual FFN)

        # ── BatchNorm: normalize across batch×spatial ─────────────────────────
        B2, N2, D2 = z.shape
        z = self.out_norm(z.reshape(B2 * N2, D2)).reshape(B2, N2, D2)

        return z                                 # [B, 256, out_dim]
