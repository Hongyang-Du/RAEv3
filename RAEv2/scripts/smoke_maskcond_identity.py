#!/usr/bin/env python3
"""Smoke test for mask conditioning (Variant A, stage1/mask_cond.py).

The one check that matters before launching: with the zero-init AdaLN head, a
mask-conditioned decoder warm-started from an UNconditional checkpoint must be
FUNCTION-IDENTICAL at step 0 (any mask, incl. all-ones and one-hot) -- otherwise
the (1+gate) parameterization is broken and the warm start silently changes the
model. Also sanity-checks the stratified sampler and the MLSCombine mask path.

CPU-runnable, no data needed:
    python scripts/smoke_maskcond_identity.py
"""
import os
import sys
import types

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, _SRC)

# stage1/__init__.py imports the full encoder stack (frozen DINOv3 etc.); this test
# only needs decoders/combine/mask_cond. Register a stub package with the right
# __path__ so submodule + relative imports work WITHOUT executing __init__.py.
_pkg = types.ModuleType("stage1")
_pkg.__path__ = [os.path.join(_SRC, "stage1")]
sys.modules["stage1"] = _pkg

import torch
from transformers import AutoConfig

torch.manual_seed(0)


def small_config():
    cfg = AutoConfig.from_pretrained("configs/decoder/ViTXL")
    cfg.hidden_size = 64
    cfg.patch_size = 16
    cfg.image_size = 64
    cfg.decoder_hidden_size = 96
    cfg.decoder_intermediate_size = 128
    cfg.decoder_num_attention_heads = 4
    cfg.decoder_num_hidden_layers = 3
    return cfg


def main():
    from stage1.decoders import GeneralDecoder
    from stage1.combine import MLSCombine
    from stage1.mask_cond import sample_stratified_masks

    K, B, N = 23, 16, 16
    cfg = small_config()

    # 1) unconditional decoder, random weights
    uncond = GeneralDecoder(cfg, num_patches=N).eval()
    sd = uncond.state_dict()

    # 2) mask-cond decoder loads the SAME weights (fresh zero-init cond modules)
    cond = GeneralDecoder(cfg, num_patches=N, mask_cond={"K": K, "d_c": 32}).eval()
    mk, uk = cond.load_state_dict(sd, strict=False)
    assert not uk, f"unexpected keys: {uk}"
    bad = [k for k in mk if not k.startswith(("mask_embedder.", "null_cond", "ada_"))]
    assert not bad, f"missing non-cond keys: {bad}"
    n_extra = sum(p.numel() for n, p in cond.named_parameters()
                  if n.startswith(("mask_embedder.", "null_cond", "ada_")))
    print(f"[load] ok: {len(mk)} fresh cond tensors, +{n_extra} params")

    # 3) step-0 function equivalence for a spread of masks
    z = torch.randn(B, N, cfg.hidden_size)
    with torch.no_grad():
        ref = uncond(z, drop_cls_token=False).logits
        masks = {
            "all-ones": torch.ones(B, K, dtype=torch.bool),
            "one-hot(l11)": torch.zeros(B, K, dtype=torch.bool).index_fill_(1, torch.tensor([10]), True),
            "random": torch.rand(B, K) > 0.5,
            "null (mask=None)": None,
        }
        for name, m in masks.items():
            out = cond(z, drop_cls_token=False, layer_mask=m).logits
            diff = (out - ref).abs().max().item()
            assert diff == 0.0, f"step-0 NOT identical for mask={name}: max|diff|={diff:.3e}"
            print(f"[identity] {name:18s} max|diff| = {diff:.1e}  OK")
        # cond_drop rows must also be identical at zero init
        drop = torch.rand(B) < 0.5
        out = cond(z, drop_cls_token=False, layer_mask=masks["random"], cond_drop=drop).logits
        assert (out - ref).abs().max().item() == 0.0, "cond_drop path not identity at step 0"
        print("[identity] cond_drop path      OK")

    # 4) stratified sampler stats
    Bs, p_drop = 30000, 0.3
    m = sample_stratified_masks(Bs, K, p_drop, full_frac=1 / 3, uniform_frac=1 / 3)
    sizes = m.sum(1)
    assert (sizes >= 1).all(), "sampler produced an empty mask"
    frac_full = (sizes == K).float().mean().item()
    cov = torch.bincount(sizes, minlength=K + 1)[1:]
    assert frac_full > 1 / 3 - 0.02, f"full group underrepresented: {frac_full:.3f}"
    assert (cov > 0).all(), f"|S| coverage holes at sizes {(cov == 0).nonzero().flatten() + 1}"
    print(f"[sampler] full={frac_full:.3f} (>=1/3), |S| covers 1..{K}, "
          f"min count over sizes = {cov.min().item()} (of {Bs})")

    # 5) MLSCombine external-mask path == manual masked mean
    comb = MLSCombine(layers=list(range(1, K + 1)), weighting="random_drop",
                      p_drop=0.3, projector="none", dim=32, out_dim=32).train()
    toks = [torch.randn(B, N, 32) for _ in range(K)]
    mm = torch.rand(B, K) > 0.5
    mm[mm.sum(1) == 0, 0] = True
    z1 = comb(toks, mask=mm)
    stk = torch.stack(toks)                                            # [K, B, N, d]
    w = mm.float().t()[:, :, None, None]
    z2 = (stk * w).sum(0) / mm.sum(1).float()[:, None, None]
    assert torch.allclose(z1, z2, atol=1e-5), \
        f"combine mask path != masked mean: {(z1 - z2).abs().max():.3e}"
    print("[combine] external-mask masked mean  OK")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
