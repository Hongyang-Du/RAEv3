"""Stage-0 denoising-AE feature fusion (DECOUPLED from the decoder).

Reframes "feature fusion" as denoising a layer-dropout-corrupted mean. The K frozen
DINOv3 layer tokens are param-free pooled (renormalized masked-mean + mask-gated
cls-surrogate) into z0 [B, N, dim]; a plain spatial-ViT encoder E compresses z0 to a
bottleneck latent z [B, N, d] via a SINGLE channel projection at its OUTPUT (all
mixing happens at full width first). The layer axis is gone before E ever runs, so E
never sees the drop mask -- it is a pure denoiser.

Two flavors, trained as SEPARATE Stage-0 runs, sharing this encoder architecture:

  MAE  : recon = D_mae(E(pool(dropped)))    vs   pool(full)     [FIXED target]
  JEPA : z_hat = P(E(pool(dropped)))        vs   E(pool(full))  [LIVE target + SIGReg]

The MAE decoder and the JEPA predictor are throwaway Stage-0 scaffolds (discarded
after Stage-0, never the pixel decoder). Only E is checkpointed for Stage-1/DiT
(no EMA -- LeJEPA). `d` is the experiment's variable: d==dim is "no compression",
down to 16 is 64x. See train_mae_denoise.py / train_jepa_denoise.py.
"""
import torch
import torch.nn as nn

from .jepa_predictor import SpatialBlock


def pool_layers(layer_tokens, mask, cls_surrogate: bool):
    """Param-free depth pooling -> z0 [B, N, dim].

    layer_tokens : list of K tensors [B, N, dim]  (frozen DINOv3 per-layer)
    mask         : [B, K] bool, True = layer kept

    Renormalized masked-mean over kept layers, then (mask-gated) L_last token-mean as a
    cls-surrogate -- identical to DepthAttnCombine's pooling (combine.py). The mask
    collapses into the mean HERE, so nothing downstream of z0 can tell WHICH layers
    were dropped; that is what makes E a pure denoiser rather than a mask-conditioned net.
    """
    stk = torch.stack(layer_tokens, dim=0)               # [K, B, N, dim]
    K, B = stk.shape[0], stk.shape[1]
    w = mask.to(stk.dtype).t()                           # [K, B]
    w = w / w.sum(0, keepdim=True).clamp_min(1e-6)
    z0 = (w.view(K, B, 1, 1) * stk).sum(0)               # [B, N, dim]
    if cls_surrogate:
        gate = mask[:, -1].to(stk.dtype).view(B, 1, 1)   # add L_last mean only if L_last kept
        z0 = z0 + gate * stk[-1].mean(dim=1, keepdim=True)
    return z0


def _make_proj(dim: int, d: int, kind: str) -> nn.Module:
    """Bottleneck readout dim -> d. 'linear' (clean latent, MAE default) or 'mlp'
    (nonlinear projector, the JEPA/SSL convention). Capacity lives in the encoder
    depth, not here -- this only picks the d-dim readout of an already-computed rep."""
    if kind == "linear":
        return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d))
    if kind == "mlp":
        return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
                             nn.LayerNorm(dim), nn.Linear(dim, d))
    raise ValueError(f"proj must be 'linear'|'mlp', got {kind!r}")


class DenoiseEncoder(nn.Module):
    """Plain spatial-ViT at full width `dim` + ONE channel bottleneck (dim -> d) at the
    output. Input is the already-pooled z0 [B, N, dim]; E never receives the drop mask.
    A single Linear readout suffices because the non-linear work is in the `depth`
    blocks before it (proj='mlp' available for a nonlinear projector)."""

    def __init__(self, dim: int = 1024, d: int = 256, depth: int = 6, n_heads: int = 8,
                 mlp_mult: int = 4, num_tokens: int = 0, proj: str = "linear"):
        super().__init__()
        self.d = d
        self.pos = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02) if num_tokens else None
        self.blocks = nn.ModuleList(
            [SpatialBlock(dim, n_heads, mlp_mult) for _ in range(depth)])
        self.proj = _make_proj(dim, d, proj)

    def forward(self, z0):                               # z0 [B, N, dim] -> z [B, N, d]
        h = z0 if self.pos is None else z0 + self.pos.to(z0.dtype)
        for blk in self.blocks:
            h = blk(h)
        return self.proj(h)


class MAEDecoder(nn.Module):
    """Throwaway Stage-0 head: lift z [B, N, d] back to full width and reconstruct the
    frozen full-pool target [B, N, dim]. The reconstruction work lives HERE (the encoder
    can be a linear readout), so keep this at least as deep as E. Discarded after
    Stage-0 -- it is NOT the pixel decoder."""

    def __init__(self, dim: int = 1024, d: int = 256, depth: int = 6, n_heads: int = 8,
                 mlp_mult: int = 4, num_tokens: int = 0):
        super().__init__()
        self.lift = nn.Linear(d, dim)
        self.pos = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02) if num_tokens else None
        self.blocks = nn.ModuleList(
            [SpatialBlock(dim, n_heads, mlp_mult) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, z):                                # z [B, N, d] -> recon [B, N, dim]
        h = self.lift(z)
        if self.pos is not None:
            h = h + self.pos.to(h.dtype)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm_out(h))


class DenoisePredictor(nn.Module):
    """Throwaway Stage-0 JEPA predictor in the BOTTLENECK space: predict z_tgt=E(full)
    from z_ctx=E(dropped). Plain spatial-ViT -- NO mask, NO extra KV conditioning. It
    only sees z_ctx and learns the average drop correction.

    Identity residual (z_hat = z_ctx + resid, resid zero-init): both are encodings of
    the same pooled mean, so at init the prediction is exactly z_ctx and the head only
    learns the correction. (The original JepaPredictor's analytic |S|/K coefficient
    assumed z_ctx was a LINEAR masked-mean; it does not hold once z_ctx passes through
    the nonlinear encoder, so identity is the right init here.)"""

    def __init__(self, d: int, depth: int = 2, n_heads: int = 8, mlp_mult: int = 2,
                 num_tokens: int = 0):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, num_tokens, d) * 0.02) if num_tokens else None
        self.blocks = nn.ModuleList(
            [SpatialBlock(d, n_heads, mlp_mult) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, d)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)                   # resid == 0 at init -> z_hat == z_ctx

    def forward(self, z_ctx):                            # z_ctx [B,N,d] -> z_hat [B,N,d]
        h = z_ctx if self.pos is None else z_ctx + self.pos.to(z_ctx.dtype)
        for blk in self.blocks:
            h = blk(h)
        return z_ctx + self.head(self.norm_out(h))


class GroundingHead(nn.Module):
    """Regress the frozen L_last pooled token [B, out_dim] from the bottleneck latent
    z [B, N, in_dim] (mean over tokens). Prices deep-layer / cls-level info explicitly so
    the compressed latent can't drop it. Dimension-flexible (in_dim=d, out_dim=dim)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), nn.GELU(),
                                 nn.Linear(out_dim, out_dim))

    def forward(self, z):                                # z [B, N, in_dim] -> [B, out_dim]
        return self.net(z.mean(1))
