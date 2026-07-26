"""Anti-collapse semantic rent for the joint fusion+decoder trainer.

The fusion latent z_b is trained jointly with the decoder; L1+LPIPS (and, in the
ablation arm, GAN) gradients are allowed into the fusion, which on their own pull
z_b toward the shallowest layer (the collapse this project spent three rounds
diagnosing). The rent counteracts that: two LINEAR heads read z_b and must predict
the standardized mean of a *strided* set of MID / DEEP frozen-encoder layers
(targets are stop-grad). Because deep semantics are a non-linear function of the
shallow features, a *linear* head can only score low if z_b itself carries the
deep subspace in linearly-readable form -- so the head is a rent that charges z_b
for dropping deep info. Keep the head weak (linear): a strong MLP would recompute
deep from a shallow z_b and hide the collapse.

Reported R_band = 1 - MSE / Var(target) is the fraction of the (standardized)
target variance z_b linearly explains -- the collapse probe and the feedback the
lambda_sem controller holds at a setpoint. It is split into R_*_full (rows whose
mask kept ALL layers = the latent the DiT will consume) and R_*_sub (subset rows =
the any-subset extrapolation guardrail).

Layer indices are positions into combine.layers; negatives count from the end
(-1 = last layer). The default deep set {-1,-3,-5,-7} is strided, not contiguous,
so the target spans the deep block with less inter-layer redundancy than a
contiguous mean.
"""

from typing import List

import torch
import torch.nn as nn


def resolve_layer_indices(idxs: List[int], K: int) -> List[int]:
    """Map possibly-negative layer positions into [0, K)."""
    out = [i % K for i in idxs]
    assert all(0 <= i < K for i in out), (idxs, K, out)
    return out


class SemanticRentHeads(nn.Module):
    def __init__(self, dim: int, K: int, mid_layers: List[int], deep_layers: List[int],
                 hidden: int = 0, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.K = K
        self.eps = eps
        self.momentum = momentum
        self.mid_idx = resolve_layer_indices(mid_layers, K)
        self.deep_idx = resolve_layer_indices(deep_layers, K)

        def _head():
            # weak by design: linear (hidden=0) unless explicitly widened
            if hidden and hidden > 0:
                return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
            return nn.Linear(dim, dim)

        self.head_mid = _head()
        self.head_deep = _head()
        # BN-style running per-channel stats to standardize each band target
        for tag in ("mid", "deep"):
            self.register_buffer(f"{tag}_mean", torch.zeros(dim))
            self.register_buffer(f"{tag}_var", torch.ones(dim))

    def _target(self, stk: torch.Tensor, idx: List[int], tag: str) -> torch.Tensor:
        """stk [K,B,N,d] -> standardized, stop-grad band target [B,N,d]."""
        t = stk[idx].mean(0)                                   # [B,N,d] strided band mean
        rm = getattr(self, f"{tag}_mean")
        rv = getattr(self, f"{tag}_var")
        if self.training:
            with torch.no_grad():
                flat = t.reshape(-1, t.shape[-1]).float()
                bm = flat.mean(0)
                bv = flat.var(0, unbiased=False)
                rm.mul_(1 - self.momentum).add_(self.momentum * bm)
                rv.mul_(1 - self.momentum).add_(self.momentum * bv)
        t = (t - rm) / (rv + self.eps).sqrt()
        return t.detach()

    def forward(self, z_b: torch.Tensor, stk: torch.Tensor, mask=None):
        """z_b [B,N,d] (grad -> fusion+heads); stk [K,B,N,d] frozen.
        mask [B,K] bool (True=kept) splits R into full vs subset rows.
        Returns (losses: {mid,deep}, metrics: {R_<band>_<full|sub>})."""
        B = z_b.shape[0]
        if mask is not None:
            is_full = mask.bool().all(dim=1)                   # [B]
        else:
            is_full = torch.ones(B, dtype=torch.bool, device=z_b.device)

        losses, metrics = {}, {}
        for tag, idx, head in (("mid", self.mid_idx, self.head_mid),
                               ("deep", self.deep_idx, self.head_deep)):
            t = self._target(stk, idx, tag)                    # [B,N,d]
            p = head(z_b)
            se = (p - t).pow(2)                                # [B,N,d]
            losses[tag] = se.mean()
            var = t.reshape(-1, t.shape[-1]).var(0, unbiased=False).mean().clamp_min(self.eps)
            se_row = se.mean(dim=(1, 2))                        # [B]
            for split, m in (("full", is_full), ("sub", ~is_full)):
                if bool(m.any()):
                    metrics[f"R_{tag}_{split}"] = (1.0 - se_row[m].mean() / var).detach()
        return losses, metrics
