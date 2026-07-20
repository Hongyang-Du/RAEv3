# raev3

## Mask-aware decoding ablation (branch `oldnorm`): Variant A vs Variant B vs anchor

Random-drop training treats the layer mask as a pure noise source: the decoder only
sees the averaged latent z and must serve every subset with one weight set (the
Prop-1 `B(qC+(1-q)G)^-1` compromise). These two variants make the decoder mask-aware,
each relaxing a different part of the pipeline. **All three runs train FROM SCRATCH
with the identical anchor recipe** (k23, p_drop=0.3, 16 epochs, batch 32/GPU,
lr 2e-4, warmup 2, disc_start 8), so each comparison differs from the anchor in
exactly one axis.

| Run | Config / launch | What changes vs anchor | Extra params |
|---|---|---|---|
| Anchor (original) | `randomdrop-plain-k23-nano-p03-oldnorm.yaml` / `nano-drop-p03` | — | — |
| **Variant A** mask conditioning | `randomdrop-plain-k23-nano-p03-maskcond-oldnorm.yaml` / `nano-drop-p03-maskcond` | decoder gets the mask as explicit input: `MaskEmbedder` (sum of kept-layer embeddings + sinusoidal \|S\|) -> zero-init AdaLN with `(1+gate)` parameterization in every ViT block; 10% CFG-style null-embedding dropout | ~2.4M (<1% of ViT-XL, shared head + per-block diag) |
| **Variant B** depth-attn fusion | `randomdrop-plain-k23-nano-p03-depthattn-oldnorm.yaml` / `nano-drop-p03-depthattn` | aggregation relaxed: latent = masked_mean + zero-init per-position depth attention over the kept layers' tokens (`stage1.combine.DepthAttnCombine`); decoder stays unconditional | ~16.8M (combine-level; intentionally trades matched-params for a higher ceiling) |

Both variants use the same **stratified mask sampling** (1/3 full feed, 1/3
\|S\|~Uniform{1..K}, 1/3 i.i.d. Bernoulli(p_drop)) instead of pure Bernoulli, so
extreme subset sizes (l11 solo, full k23) get real training mass. The zero-init
construction means both variants start as the exact anchor function and learn their
correction from identity (also enables warm-starting from anchor ckpts via
`training.init_from`, verified but not used in this from-scratch comparison).

### Launch (Pluto)

```bash
bash run_pluto_decoder_4node.sh nano-drop-p03            # anchor
bash run_pluto_decoder_4node.sh nano-drop-p03-maskcond   # Variant A
bash run_pluto_decoder_4node.sh nano-drop-p03-depthattn  # Variant B
```

### Eval protocol

Iron rule: the conditioning/fusion mask must equal the mask that built z.
`src/eval_recon_subset.py` auto-detects mask-cond ckpts (decoder state contains
`mask_embedder.*`) and passes the matched k-hot mask for any `--idx` subset;
`DepthAttnCombine` derives its mask from `idx` internally. Per feed (k23 full,
k7 prefix, l11 solo):

```bash
python src/eval_recon_subset.py --config <train yaml> --ckpt <ckpt> --tag k23full
python src/eval_recon_subset.py --config <train yaml> --ckpt <ckpt> --idx 0,...,6 --tag k7
python src/eval_recon_subset.py --config <train yaml> --ckpt <ckpt> --idx 10 --tag l11
# Variant A only: net conditioning contribution (null-embedding A/B)
python src/eval_recon_subset.py --config <train yaml> --ckpt <ckpt> --null-cond --tag k23full_null
```

During training, Variant A additionally logs `val/psnr_null` next to `val/psnr`
(the gap = conditioning's net contribution, tracked every epoch in wandb).

Expected: full k23 feed improves most (unconditional decoding is farthest from the
full-readout specialist there), l11 rises from stratified sampling, k7 roughly flat.
Variant B has the higher ceiling on full feed (uncompressed token-level access);
its gate-collapse risk (Fig 2) is held off by random drop itself plus the 8 GAN-free
warmup epochs.

### Verification

CPU smoke tests, no data needed (both must print `ALL CHECKS PASSED`):

```bash
python scripts/smoke_maskcond_identity.py    # A: zero-init => bit-exact identity, sampler, grad flow
python scripts/smoke_depthattn_identity.py   # B: fusion no-op, anchor-semantics drift <1e-5, padding leak = 0
```
