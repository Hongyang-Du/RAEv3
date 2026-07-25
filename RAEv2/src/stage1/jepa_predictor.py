"""Stage-0 JEPA scaffolding: a throwaway predictor + grounding head used ONLY to
pretrain DepthAttnCombine with a self-supervised layer-JEPA objective
(train_fusion_jepa.py). Neither module is part of the decoder-facing latent path:
the decoder/DiT consume the RAW combine output z, so any alignment ability these
modules learn stays OUT of the latent. Both are discarded after Stage-0.

JEPA correspondence (view axis = DEPTH, not space): frozen DINOv3 is the shared
featurizer; DepthAttnCombine is the shared encoder head; x-view = a random layer
subset (ctx), y-view = the full layer set (target); no EMA target-encoder (LeJEPA /
DINO multi-crop shared-encoder form). This predictor is the JEPA predictor.
"""
import torch
import torch.nn as nn

from .mask_cond import MaskEmbedder


class SpatialBlock(nn.Module):
    """Pre-norm ViT block with self-attention over the N spatial tokens."""

    def __init__(self, dim: int, n_heads: int, mlp_mult: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_mult * dim), nn.GELU(),
                                 nn.Linear(mlp_mult * dim, dim))

    def forward(self, x):                            # x [B, N, d]
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.norm2(x))


class JepaPredictor(nn.Module):
    """Predict the full-set latent z_full from a subset latent z_ctx + the drop mask.

    Three deliberate choices (Stage-0 spec):
      * The mask (WHICH layers are missing) is an EXPLICIT side input, never something
        z_ctx must smuggle -> z_ctx stays pure content (kills the "encode subset id in
        the latent" cheat), and the analytic |S|/K scale is handed over for free.
      * Spatial self-attention lives HERE, not in the per-position encoder: a dropped
        layer's content at position n correlates with kept-layer tokens at neighbouring
        positions (DINOv3 features are spatially smooth); the predictor exploits that
        prior and it is thrown away with the predictor, never touching the latent.
      * Analytic residual. With attention-correction = 0 (masked-mean base):
            z_full = (|S|/K)*z_ctx + (1/K)*sum_{i not in S} h_i
        so the closed-form (|S|/K)*z_ctx term is added back explicitly and the
        zero-init head only learns the genuinely-unknown missing-layer sum. At step 0
        the prediction is exactly (|S|/K)*z_ctx, mirroring DepthAttn's zero-init
        identity (the |S|/K coeff is an inductive bias, exact only at init; the head
        absorbs the drift once the attention correction turns on).
    """

    def __init__(self, dim: int, K: int, n_layers: int = 2, n_heads: int = 8,
                 mlp_mult: int = 2, num_tokens: int = 0):
        super().__init__()
        self.K = K
        self.mask_emb = MaskEmbedder(K, dim)                     # k-hot mask -> c [B, dim]
        # Learned spatial position embedding. The whole reason the predictor gets spatial
        # attention is to use the "neighbouring kept-layer token" prior for the dropped
        # layer; explicit positions make that addressing easy instead of leaning on the
        # position info residual in DINOv3 features. num_tokens=0 -> disabled. Created at
        # init (not lazily) so it is registered in the optimizer.
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02) if num_tokens else None
        self.blocks = nn.ModuleList(
            [SpatialBlock(dim, n_heads, mlp_mult) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)                           # residual == 0 at init

    def forward(self, z_ctx, mask):                              # z_ctx [B,N,d] mask [B,K] bool
        B, N, d = z_ctx.shape
        c = self.mask_emb(mask.to(z_ctx.dtype)).unsqueeze(1)     # [B, 1, d]
        h = z_ctx + c                                            # inject mask cond per token
        if self.pos_emb is not None:
            h = h + self.pos_emb.to(h.dtype)
        for blk in self.blocks:
            h = blk(h)
        resid = self.head(self.norm_out(h))                      # [B,N,d], 0 at init
        coef = (mask.to(z_ctx.dtype).sum(-1) / self.K).view(B, 1, 1)   # |S|/K
        return coef * z_ctx + resid


class GroundingHead(nn.Module):
    """Regress the frozen L_last pooled token (the deep-layer / CLS-surrogate signal)
    from the subset latent z_ctx. This is what prices deep-layer info explicitly:
    when the deep layer is dropped from z_ctx, z_ctx must still carry enough to
    predict it -> the shallow-collapse incentive is countered at the objective level,
    not by architecture surgery."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
                                 nn.Linear(dim, dim))

    def forward(self, z):                                        # z [B,N,d] -> [B,d]
        return self.net(z.mean(1))
