# Decoupled stage-2 evaluation: dFID + latent-space metrics

Standard gFID mixes two failure modes: the stage-1 reconstruction bottleneck
(decoder blur, color shift, information loss) and the stage-2 fit (how well the
DiT matches the latent distribution). `eval_latent_dit.py` +
`eval/latent_metrics.py` separate them, so a change to the autoencoder can be
attributed to "better/worse reconstruction" vs "easier/harder latent space for
diffusion", and a gap can be attributed to stage-2 underfitting vs decoder
manifold roughness.

## The pixel-space triangle

With `real` = N real images, `recon` = D(E(real)), `gen` = D(Z_sampled):

| Metric | Compares | Measures |
|---|---|---|
| `rfid` | recon vs real | stage-1 reconstruction bottleneck only |
| `gfid` | gen vs real | the standard number (both errors mixed) |
| `dfid` | gen vs recon | **decoder-decoupled** generation error |

Both sides of `dfid` pass through the identical map (Inception ∘ Decoder), so
the decoder's systematic losses cancel; what remains is the latent-distribution
mismatch as seen through the decoder. Under the Gaussian approximation FID is a
squared Fréchet (2-Wasserstein) distance, so the triangle inequality gives

```
sqrt(gfid) <= sqrt(rfid) + sqrt(dfid)
```

which the script prints as a sanity check. Two stage-1 variants with similar
`rfid` but very different `dfid` (same stage-2 recipe) differ in how
diffusion-friendly their latent manifold is, not in reconstruction quality.

Note `dfid` must always be read next to `rfid`: a decoder that collapses
diversity pulls recon and gen onto the same degenerate manifold and makes
`dfid` look good for the wrong reason.

## Latent-space metrics

Computed between real latents E(X) and sampled latents Z in the normalized
space the DiT runs in ([N, 1024, 16, 16]). There is no Inception here — the
latents are the features — so dimensionality has to be handled explicitly
(never estimate a covariance on the flattened 262k-dim latent).

| Metric | Samples x dim | Sees | Blind to |
|---|---|---|---|
| `per_token_fd` | N·256 x 1024 | marginal token distribution, all positions pooled | spatial structure; position-specific failures (diluted) |
| `per_position_fd` (16x16 heatmap) | N x 1024 per position | where in space the fit fails (edges/corners differ a lot from center tokens) | cross-token joint structure |
| `pooled_fd` | N x 1024 | image-level semantic distribution (spatial mean-pool) | local detail |
| `rp_fd`, `rp_rbf_mmd2` | N x 2048 (fixed-seed Gaussian projection of the full flattened latent) | **cross-token joint structure** | — (MMD also sees beyond two moments) |

All FD variants assume Gaussianity (two moments). The RBF-MMD is the guard
against a sampler that matches mean/covariance but misses mode structure —
"latent distribution is fit" should mean **all** of these are low, not just one.

## Reading the numbers

| latent metrics | dfid | diagnosis |
|---|---|---|
| low | low | stage-2 fits, decoder manifold is smooth |
| high | high | stage-2 underfitting — fix the diffusion side first |
| low | high | latent distribution is matched but the decoder amplifies small OOD deviations: **decoder manifold roughness** — target with stage-1 regularization |

The `per_position_fd` heatmap localizes the "high latent" case (e.g. border
tokens unfit while center tokens are fine).

## Usage

```bash
# numerical self-check of the metric library (analytic FD ground truths etc.)
python tests/test_latent_metrics.py cuda

# full evaluation, single GPU
python src/eval_latent_dit.py \
    --config configs/stage2/training/<run>.yaml \
    --ckpt   ckpts_full/stage2/<run>/checkpoints/ep-XXXX.pt \
    --data   /datasets/imagenet-256-full \
    --num-samples 10000 \
    --real-cache ckpts_full/eval_cache/real10k_seed42.npz \
    --out    ckpts_full/stage2/<run>/latent_eval.json
```

Outputs: `<out>.json` (rfid/gfid/dfid/IS + all latent metrics),
`<out>_pos_fd.npy` and `_pos_fd.png` (per-position heatmap).

Useful flags:

- `--real-cache <npz>` — caches real images + real latents + reconstructions;
  reused across stage-2 checkpoints of the same stage-1 (the cache stores a
  stage-1 config fingerprint and rebuilds automatically if it changes).
- `--skip-pixel` — latent metrics only, no Inception passes (fast sweeps).
- `--save-gen <npz>` — dump sampled latents + decoded images for offline use.
- `--ref-npz` — use a fixed real set (e.g. val 50k npz) instead of sampling
  train images.
- Guidance flags (`--cfg-scale`, `--ig-scale`, ...) identical to
  `eval_fid_dit.py`; sampling path (Euler ODE, bf16 autocast) is identical too,
  so `gfid` here matches `eval_fid_dit.py` at the same samples/seed/steps.

## Caveats

- **Fixed-N comparisons only.** Every FD/FID/MMD here is biased in sample
  count; compare runs only at identical `--num-samples`, `--seed`,
  `--proj-dim`.
- **Latent metrics are not comparable across different stage-1 models.** Each
  stage-1 defines its own latent space (dimension, scale, normalization), so
  absolute latent-FD values only rank stage-2 checkpoints / samplers / guidance
  settings *within* one latent space. Across stage-1 variants, compare `dfid`
  (pixel space is shared).
- Latent metrics measure the stage-2 fit; they do **not** measure manifold
  smoothness directly. For a diffusion-free probe of decoder roughness, sweep
  FID(D(E(x)), D(E(x)+σε)) over σ, or measure decoder Jacobian norms /
  interpolation LPIPS — these isolate the decoder entirely.
