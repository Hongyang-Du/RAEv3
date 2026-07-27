"""3-band depth-JEPA heads for the DECODER-LESS Stage-0 fusion trainer.

This is the semantic-rent idea (src/stage1/semantic_rent.py) PROMOTED from an
auxiliary anti-collapse tax to the WHOLE training objective. There is no decoder
and no pixel/GAN/LPIPS loss: the DepthAttnCombine fusion latent z_b is trained
ONLY to linearly predict the standardized, stop-grad mean of a set of strided
frozen-encoder LAYER bands (shallow / mid / deep by default). Because the layers
are randomly dropped per sample (stratified mask), a band whose layers were
dropped from the input must be *imputed* from the surviving layers -> this is
masked modeling along the DEPTH axis, in latent space against frozen targets
(JEPA-flavoured, NOT pixel-MAE).

Why LINEAR heads (hidden=0, the default): a linear head can only score low if
z_b itself carries each band's subspace in linearly-readable form. That is
exactly the property that (a) prevents the shallow-layer collapse (the deep head
charges z_b for dropping deep info) and (b) makes z_b a well-conditioned,
linearly-structured latent for a downstream DiT. A strong MLP head would
recompute deep semantics from a lazy z_b and hide the collapse -- do NOT widen
the heads to chase a lower loss. See the rent lineage in semantic_rent.py.

R_band = 1 - MSE / Var(target): the fraction of the (standardized) band-target
variance z_b linearly explains. Split into R_<band>_full (rows whose mask kept
ALL layers = the latent the DiT will consume) and R_<band>_sub (subset rows =
the depth-imputation / any-subset guardrail).
"""

from typing import Dict, List

import torch
import torch.nn as nn

from .semantic_rent import resolve_layer_indices


class BandPredictHeads(nn.Module):
    """One weak (linear by default) head per band; each predicts the standardized,
    stop-grad mean of its strided frozen-layer set from the fusion latent z_b.

    bands   : {name -> [layer positions into combine.layers; negatives from end]}
    weights : {name -> loss weight}; any band missing here defaults to 1.0.
    """

    def __init__(self, dim: int, K: int, bands: Dict[str, List[int]],
                 weights: Dict[str, float] = None, hidden: int = 0,
                 momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        assert len(bands) > 0, "need at least one band"
        self.K = K
        self.eps = eps
        self.momentum = momentum
        self.names = list(bands.keys())                       # stable order
        self.band_idx = {n: resolve_layer_indices(list(idxs), K) for n, idxs in bands.items()}
        weights = dict(weights or {})
        self.weights = {n: float(weights.get(n, 1.0)) for n in self.names}

        def _head():
            # weak by design: linear (hidden=0) unless explicitly widened
            if hidden and hidden > 0:
                return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
            return nn.Linear(dim, dim)

        self.heads = nn.ModuleDict({n: _head() for n in self.names})
        # BN-style running per-channel stats to standardize each band target
        for n in self.names:
            self.register_buffer(f"{n}_mean", torch.zeros(dim))
            self.register_buffer(f"{n}_var", torch.ones(dim))

    def _target(self, stk: torch.Tensor, name: str) -> torch.Tensor:
        """stk [K,B,N,d] -> standardized, stop-grad band target [B,N,d]."""
        t = stk[self.band_idx[name]].mean(0)                  # [B,N,d] strided band mean
        rm = getattr(self, f"{name}_mean")
        rv = getattr(self, f"{name}_var")
        if self.training:
            with torch.no_grad():
                flat = t.reshape(-1, t.shape[-1]).float()
                rm.mul_(1 - self.momentum).add_(self.momentum * flat.mean(0))
                rv.mul_(1 - self.momentum).add_(self.momentum * flat.var(0, unbiased=False))
        t = (t - rm) / (rv + self.eps).sqrt()
        return t.detach()

    def forward(self, z_b: torch.Tensor, stk: torch.Tensor, mask=None):
        """z_b [B,N,d] (grad -> fusion+heads); stk [K,B,N,d] frozen (stop-grad).
        mask [B,K] bool (True=kept) splits R into full vs subset rows.

        Returns (loss: scalar weighted sum, parts: {band -> mse}, metrics:
        {R_<band>_<full|sub>}). The BN-style target standardization means each
        band contributes an O(1) MSE, so the weights are the only knob."""
        B = z_b.shape[0]
        if mask is not None:
            is_full = mask.bool().all(dim=1)                  # [B]
        else:
            is_full = torch.ones(B, dtype=torch.bool, device=z_b.device)

        total = z_b.new_zeros(())
        parts, metrics = {}, {}
        for name in self.names:
            t = self._target(stk, name)                       # [B,N,d]
            p = self.heads[name](z_b)
            se = (p - t).pow(2)                               # [B,N,d]
            parts[name] = se.mean()
            total = total + self.weights[name] * parts[name]
            var = t.reshape(-1, t.shape[-1]).var(0, unbiased=False).mean().clamp_min(self.eps)
            se_row = se.mean(dim=(1, 2))                       # [B]
            for split, m in (("full", is_full), ("sub", ~is_full)):
                if bool(m.any()):
                    metrics[f"R_{name}_{split}"] = (1.0 - se_row[m].mean() / var).detach()
        return total, parts, metrics
