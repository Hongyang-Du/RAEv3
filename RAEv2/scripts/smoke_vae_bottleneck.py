#!/usr/bin/env python3
"""Smoke test for the VAE-bottleneck combine (VAEBottleneck, MLSCombine projector="vae",
and the CompressedCombine two-stage wrapper).

Checks before launching:
  1. width schedule: auto-derived halving lands exactly on out_dim; explicit
     `widths=` override is honored as-is
  2. shapes: dim -> out_dim (out_dim << dim) through MLSCombine end-to-end
  3. variational=False: deterministic (mu only, no logvar, no noise) regardless of
     train/eval
  4. variational=True: logvar == 0 at init (zero-init to_logvar -> sigma=1), train-mode
     forward is stochastic (reparameterized) across calls, eval-mode is deterministic
     (posterior mean), and probe/eval subsets (idx=...) skip reparam even in train mode
     (mirrors MLSCombine._noise's existing idx-skip convention)
  5. last_mu / last_logvar are stashed on the module with the right shape (the hook
     train_decoder.py's KL term reads)
  6. grad flow: every bottleneck param (encoder stages + to_logvar) is in the graph
  7. CompressedCombine(inner=DepthAttnCombine): the inner combine is untouched (still
     bit-exact zero-init identity, same check as smoke_depthattn_identity.py), shapes
     work end-to-end, p_drop/layers/K proxy to the inner combine, and grad flows to
     BOTH the inner fusion params and the bottleneck params

CPU-runnable, no data needed:
    python scripts/smoke_vae_bottleneck.py
"""
import os
import sys
import types

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, _SRC)

# bypass stage1/__init__.py (imports the full encoder stack); see smoke_maskcond_identity.py
_pkg = types.ModuleType("stage1")
_pkg.__path__ = [os.path.join(_SRC, "stage1")]
sys.modules["stage1"] = _pkg

import torch

torch.manual_seed(0)


def main():
    from stage1.combine import VAEBottleneck, MLSCombine

    # 1) width schedule ---------------------------------------------------------
    auto = VAEBottleneck(dim=1024, out_dim=4)
    assert auto.widths == [1024, 512, 256, 128, 64, 32, 16, 8, 4], auto.widths
    print(f"[widths] auto-derived 1024->4: {auto.widths}  OK")

    custom = [1024, 512, 256, 128, 64, 32, 16, 4]           # user's literal schedule (16->4 = /4)
    manual = VAEBottleneck(dim=1024, out_dim=4, widths=custom)
    assert manual.widths == custom
    print(f"[widths] explicit override honored: {manual.widths}  OK")

    # 2) shapes, deterministic (variational=False) -------------------------------
    B, N, dim, out_dim = 4, 16, 64, 4
    det = VAEBottleneck(dim=dim, out_dim=out_dim, variational=False)
    z0 = torch.randn(B, N, dim)
    mu, logvar = det(z0)
    assert mu.shape == (B, N, out_dim) and logvar is None
    mu2, _ = det(z0)
    assert torch.equal(mu, mu2), "deterministic bottleneck must be exactly repeatable"
    print(f"[shape/det] {tuple(z0.shape)} -> {tuple(mu.shape)}, repeatable  OK")

    # 3) variational=True: logvar==0 at init (zero-init to_logvar -> sigma=1) ----
    var = VAEBottleneck(dim=dim, out_dim=out_dim, variational=True)
    mu, logvar = var(z0)
    assert logvar is not None and (logvar == 0).all(), "to_logvar must be zero-init"
    print("[variational] logvar == 0 at init (sigma=1)  OK")

    # 4) MLSCombine end-to-end: weighting=mean removes layer-drop randomness so the
    #    ONLY stochastic source under test is the VAE reparameterization itself.
    K = 23
    toks = [torch.randn(B, N, dim) for _ in range(K)]

    comb_det = MLSCombine(layers=list(range(1, K + 1)), weighting="mean",
                          cls_surrogate=True, projector="vae", variational=False,
                          dim=dim, out_dim=out_dim)
    comb_det.train()
    z1 = comb_det(toks)
    z2 = comb_det(toks)
    assert z1.shape == (B, N, out_dim)
    assert torch.equal(z1, z2), "variational=False must be deterministic even in train()"
    assert comb_det.last_logvar is None
    print(f"[combine/det] shape={tuple(z1.shape)}, train-mode repeatable, last_logvar=None  OK")

    comb_var = MLSCombine(layers=list(range(1, K + 1)), weighting="mean",
                          cls_surrogate=True, projector="vae", variational=True,
                          dim=dim, out_dim=out_dim)
    comb_var.train()
    z1 = comb_var(toks)
    z2 = comb_var(toks)
    assert not torch.equal(z1, z2), "variational=True must reparameterize (stochastic) in train()"
    assert comb_var.last_mu.shape == (B, N, out_dim) and comb_var.last_logvar.shape == (B, N, out_dim)
    comb_var.eval()
    z3 = comb_var(toks)
    z4 = comb_var(toks)
    assert torch.equal(z3, z4), "eval() must be deterministic (posterior mean, no reparam noise)"
    print("[combine/vae] train-mode stochastic, eval-mode deterministic, last_mu/logvar shaped OK  OK")

    # idx (probe/eval subset) must skip reparam even while comb_var.training is True,
    # mirroring MLSCombine._noise's existing idx-skip convention (LOO/solo probes must
    # be repeatable from a single trained model).
    comb_var.train()
    zp1 = comb_var(toks, idx=[0, 1, 2])
    zp2 = comb_var(toks, idx=[0, 1, 2])
    assert torch.equal(zp1, zp2), "idx-subset probe path must be deterministic regardless of train()"
    print("[combine/vae] idx-subset probe path deterministic under train()  OK")

    assert comb_var.has_params, "VAE bottleneck combine must report has_params=True (DDP-wrappable)"
    print("[combine/vae] has_params=True  OK")

    # 5) grad flow: every bottleneck param reachable from a fresh module ----------
    comb_fresh = MLSCombine(layers=list(range(1, K + 1)), weighting="mean",
                            cls_surrogate=True, projector="vae", variational=True,
                            dim=dim, out_dim=out_dim).train()
    z = comb_fresh(toks)
    z.sum().backward()
    missing = [n for n, p in comb_fresh.named_parameters() if p.grad is None]
    assert not missing, f"no grad for {missing}"
    print("[grad-flow] every bottleneck param in the graph  OK")

    # 6) CompressedCombine wrapping DepthAttnCombine (the "two-stage" design: inner
    #    fusion untouched, bottleneck chained after its completed dim-wide output) --
    from stage1.combine import DepthAttnCombine, CompressedCombine

    inner_cfg = dict(target="stage1.combine.DepthAttnCombine",
                     params=dict(layers=list(range(1, K + 1)), p_drop=0.3, cls_surrogate=True,
                                 dim=dim, out_dim=dim, n_layers=2, n_heads=4, mlp_mult=2))
    wrapped = CompressedCombine(inner=inner_cfg, dim=dim, out_dim=out_dim, variational=True)

    # 6a) the inner DepthAttnCombine is untouched: at zero-init it must still be
    #     bit-exact identical to the plain masked mean (same check as
    #     smoke_depthattn_identity.py), proving the wrapper didn't alter its forward.
    def manual_masked_mean(toks, mask, cls_surrogate):
        stk = torch.stack(toks)
        Kk, Bb = stk.shape[0], stk.shape[1]
        w = mask.to(stk.dtype).t()
        w = w / w.sum(0, keepdim=True).clamp_min(1e-6)
        z = (w.view(Kk, Bb, 1, 1) * stk).sum(0)
        if cls_surrogate:
            z = z + stk[-1].mean(dim=1, keepdim=True)
        return z

    wrapped.eval()
    ref = manual_masked_mean(toks, torch.ones(B, K, dtype=torch.bool), True)
    diff = (wrapped.inner(toks) - ref).abs().max().item()
    assert diff == 0.0, f"wrapping changed the inner DepthAttnCombine's own forward: {diff:.3e}"
    print(f"[wrapped] inner DepthAttnCombine still zero-init identity, max|diff|={diff:.1e}  OK")

    # 6b) end-to-end shape through both stages
    wrapped.train()
    z = wrapped(toks)
    assert z.shape == (B, N, out_dim)
    print(f"[wrapped] DepthAttnCombine -> VAEBottleneck shape {tuple(z.shape)}  OK")

    # 6c) p_drop / layers / K proxy to the inner combine (trainer mutates
    #     combine.p_drop directly; other code reads combine.layers/.K)
    wrapped.p_drop = 0.7
    assert wrapped.inner.p_drop == 0.7, "p_drop setter must proxy to inner"
    assert wrapped.p_drop == 0.7 and wrapped.layers == list(range(1, K + 1)) and wrapped.K == K
    print("[wrapped] p_drop/layers/K proxy to inner combine  OK")

    assert wrapped.has_params
    print("[wrapped] has_params=True  OK")

    # 6d) grad flow reaches BOTH the inner fusion params and the bottleneck params
    z = wrapped(toks)
    z.sum().backward()
    missing = [n for n, p in wrapped.named_parameters() if p.grad is None]
    assert not missing, f"no grad for {missing}"
    n_inner = sum(1 for n, _ in wrapped.named_parameters() if n.startswith("inner."))
    n_bneck = sum(1 for n, _ in wrapped.named_parameters() if n.startswith("bottleneck."))
    assert n_inner > 0 and n_bneck > 0
    print(f"[wrapped] grad flows to inner ({n_inner} tensors) and bottleneck ({n_bneck} tensors)  OK")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
