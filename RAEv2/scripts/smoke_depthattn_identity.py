#!/usr/bin/env python3
"""Smoke test for Variant B (DepthAttnCombine, stage1/combine.py).

Checks before launching:
  1. step-0 identity: zero-init fusion -> output == masked mean (+ cls surrogate),
     EXACTLY, for external / internal / idx / eval-full masks
  2. warm-start: loading the anchor's EMPTY (param-free) combine state_dict leaves
     only fusion.* params fresh
  3. key_padding correctness: with a NON-zero fusion, perturbing a DROPPED layer's
     tokens must not change the output; perturbing a KEPT layer must
  4. grad flow: every fusion param is in the graph (DDP safety)

CPU-runnable, no data needed:
    python scripts/smoke_depthattn_identity.py
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


def manual_masked_mean(toks, mask, cls_surrogate):
    # EXACTLY the reduction order of DepthAttnCombine/MLSCombine._mix (normalize
    # weights first, then weighted sum) so the identity check can demand bit-equality.
    stk = torch.stack(toks)                                       # [K, B, N, d]
    K, B = stk.shape[0], stk.shape[1]
    w = mask.to(stk.dtype).t()
    w = w / w.sum(0, keepdim=True).clamp_min(1e-6)
    z = (w.view(K, B, 1, 1) * stk).sum(0)
    if cls_surrogate:
        z = z + stk[-1].mean(dim=1, keepdim=True)
    return z


def main():
    from stage1.combine import DepthAttnCombine

    K, B, N, d = 23, 8, 16, 64
    comb = DepthAttnCombine(layers=list(range(1, K + 1)), p_drop=0.3,
                            cls_surrogate=True, dim=d, out_dim=d,
                            n_layers=2, n_heads=4, mlp_mult=2)
    toks = [torch.randn(B, N, d) for _ in range(K)]

    # 1) step-0 identity for every mask path
    with torch.no_grad():
        comb.train()
        m = torch.rand(B, K) > 0.5
        m[m.sum(1) == 0, 0] = True
        cases = {
            "external mask": (dict(mask=m), m),
            "idx subset": (dict(idx=[10]), torch.zeros(B, K, dtype=torch.bool).index_fill_(1, torch.tensor([10]), True)),
        }
        comb.eval()
        cases["eval full"] = (dict(), torch.ones(B, K, dtype=torch.bool))
        for name, (kw, mm) in cases.items():
            z = comb(toks, **kw)
            ref = manual_masked_mean(toks, mm, True)
            diff = (z - ref).abs().max().item()
            assert diff == 0.0, f"step-0 NOT identical ({name}): max|diff|={diff:.3e}"
            print(f"[identity] {name:14s} max|diff| = {diff:.1e}  OK")
        # internal training sampler runs (stochastic; just check shape + finite)
        comb.train()
        z = comb(toks)
        assert z.shape == (B, N, d) and torch.isfinite(z).all()
        print("[identity] internal-sample path runs, finite  OK")

        # cross-check vs the anchor's MLSCombine semantics (its idx path uses
        # sub.mean(0), a different-but-equal reduction order -> float-eps only)
        from stage1.combine import MLSCombine
        anchor = MLSCombine(layers=list(range(1, K + 1)), weighting="random_drop",
                            p_drop=0.3, cls_surrogate=True, projector="none",
                            dim=d, out_dim=d).eval()
        comb.eval()
        for name, kw in [("full", {}), ("idx=[10]", {"idx": [10]}),
                         ("idx=0..6", {"idx": list(range(7))})]:
            drift = (comb(toks, **kw) - anchor(toks, **kw)).abs().max().item()
            assert drift < 1e-5, f"anchor-semantics drift too large ({name}): {drift:.3e}"
            print(f"[anchor-x] {name:9s} vs MLSCombine drift = {drift:.1e}  (<1e-5 OK)")

    # 2) warm-start from a param-free-combine ckpt (empty state_dict)
    mk, uk = comb.load_state_dict({}, strict=False)
    assert not uk and all(k.startswith("fusion.") for k in mk), f"unexpected={uk} missing={mk}"
    n_fusion = sum(p.numel() for n, p in comb.named_parameters() if n.startswith("fusion."))
    print(f"[warmstart] empty ckpt load ok: {len(mk)} fusion tensors fresh (+{n_fusion / 1e3:.0f}K params @ d={d})")

    # 3) padding correctness with a NON-zero fusion
    with torch.no_grad():
        for p in comb.fusion.parameters():
            p.add_(torch.randn_like(p) * 0.05)                    # break zero-init
        comb.eval()
        m = torch.ones(B, K, dtype=torch.bool)
        m[:, 5] = False                                           # drop layer position 5
        z_ref = comb(toks, mask=m)
        toks_pert = [t.clone() for t in toks]
        toks_pert[5] = toks_pert[5] + 100.0                       # perturb the DROPPED layer
        z_pert = comb(toks_pert, mask=m)
        diff_drop = (z_pert - z_ref).abs().max().item()
        assert diff_drop == 0.0, f"dropped layer leaked through attention: {diff_drop:.3e}"
        toks_pert2 = [t.clone() for t in toks]
        toks_pert2[6] = toks_pert2[6] + 1.0                       # perturb a KEPT layer
        diff_keep = (comb(toks_pert2, mask=m) - z_ref).abs().max().item()
        assert diff_keep > 0, "kept layer had no influence?!"
        print(f"[padding] dropped-layer leak = {diff_drop:.1e} (OK), kept-layer effect = {diff_keep:.2e} (OK)")

    # 4) grad flow to every fusion param (fresh zero-init module, worst case)
    comb2 = DepthAttnCombine(layers=list(range(1, K + 1)), p_drop=0.3,
                             cls_surrogate=True, dim=d, out_dim=d,
                             n_layers=2, n_heads=4, mlp_mult=2).train()
    z = comb2(toks, mask=m)
    z.sum().backward()
    missing = [n for n, p in comb2.named_parameters() if p.grad is None]
    assert not missing, f"no grad for {missing}"
    print("[grad-flow] every fusion param in the graph  OK")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
