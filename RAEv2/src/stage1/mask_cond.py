"""Mask conditioning (Variant A) for the random-drop MLS decoder.

The random-drop trainer averages a per-sample layer subset S into z; without
conditioning the decoder must serve every subset with one weight set (the
Prop-1 B(qC+(1-q)G)^-1 compromise). Here the mask becomes an explicit decoder
input: MaskEmbedder turns the k-hot mask m into a condition vector c that the
decoder injects via AdaLN (see decoders/decoder.py), so it can realize
per-subset decoders D_S = B_S C_S^-1. The (1+gate) AdaLN parameterization is
zero-init -> at step 0 the conditioned decoder is function-identical to the
unconditional checkpoint it warm-starts from.

Also provides the stratified mask sampler: i.i.d. Bernoulli concentrates |S|
at the binomial mean, so extreme subset sizes (|S|=1, |S|=K) are never really
trained; the stratified mix (full / uniform-|S| / bernoulli) covers them.
"""

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """DiT-style timestep embedding for scalar t [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32,
                                                           device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class MaskEmbedder(nn.Module):
    """k-hot layer mask m [B, K] -> condition vector c [B, d_c].

    Sum (not mean) over kept-layer embeddings so c carries WHICH layers and HOW
    MANY; the explicit sinusoidal |S| embedding + LayerNorm stabilize the ~20x
    norm gap between |S|=1 and |S|=K."""

    def __init__(self, K: int, d_c: int):
        super().__init__()
        self.K = K
        self.d_c = d_c
        self.layer_emb = nn.Parameter(torch.randn(K, d_c) * 0.02)
        self.mlp = nn.Sequential(nn.LayerNorm(d_c),
                                 nn.Linear(d_c, d_c), nn.SiLU(),
                                 nn.Linear(d_c, d_c))

    def forward(self, m: torch.Tensor) -> torch.Tensor:          # m: [B, K] bool/float
        m = m.to(self.layer_emb.dtype)
        c = m @ self.layer_emb                                   # [B, d_c]
        c = c + sinusoidal_embedding(m.sum(-1), self.d_c)
        return self.mlp(c)


def sample_stratified_masks(B: int, K: int, p_drop: float,
                            full_frac: float = 1 / 3, uniform_frac: float = 1 / 3,
                            device=None) -> torch.Tensor:
    """Stratified per-sample layer masks [B, K] bool (True = layer kept).

    Per-sample group assignment (random, so every rank/micro-batch mixes):
      full_frac     m = all-ones (the sandwich full branch)
      uniform_frac  |S| ~ Uniform{1..K}, then a uniform subset of that size
                    -> extreme sizes (1, K) get real training mass
      remainder     i.i.d. Bernoulli(keep = 1-p_drop), >=1 kept
                    (keeps the correspondence with the theory analysis)
    """
    u = torch.rand(B, device=device)
    is_full = u < full_frac
    is_unif = (~is_full) & (u < full_frac + uniform_frac)

    # bernoulli base (also the fallback the other groups overwrite)
    keep = torch.rand(B, K, device=device) > p_drop
    dead = ~keep.any(1)
    if dead.any():                                   # always keep >= 1 layer
        keep[dead, torch.randint(K, (int(dead.sum()),), device=device)] = True

    # uniform-|S| group: rank the per-row random scores, keep the top-s
    if is_unif.any():
        n = int(is_unif.sum())
        s = torch.randint(1, K + 1, (n,), device=device)              # [n] in {1..K}
        scores = torch.rand(n, K, device=device)
        rank = scores.argsort(dim=1).argsort(dim=1)                   # 0..K-1 per row
        keep[is_unif] = rank < s[:, None]                             # top-s kept

    keep[is_full] = True
    return keep
