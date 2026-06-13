# Config-Driven Stage-1 Decoder + E2E Training — Design

**Date:** 2026-06-12
**Status:** Approved (pending spec review)

## Problem

Stage-1 decoder retraining lives in **9 near-duplicate scripts**
(`src/train_decoder_mls*.py`, ~540–700 lines each). They share ~80% boilerplate
(DDP, data loader, DINO discriminator + adaptive weight, LPIPS, EMA,
checkpoint/resume, val PSNR, LOO/solo probes, logging). The diff between the
LN-dropmean and BN-dropmean scripts is only 67 lines out of 541 — almost
entirely the projector class plus a few naming strings. Each new experiment is
a copy-paste fork, and every cross-cutting fix (e.g. the val-1k PSNR/SSIM block)
must be applied 9 times by hand.

Stage-2 DiT training is already config-driven (`configs/stage2/training/*.yaml`
+ `src/train.py`, OmegaConf structured config with `target:`/`params:`
`instantiate_from_config`). The official stage-1 trainer
(`src/train_stage1.py`) follows the same pattern but uses the stock
`stage1.RAE` — it does NOT cover the custom MLS-combine / projector / SIGReg /
dropmean recipe the user's scripts implement.

## Goal

One config-driven stage-1 decoder trainer + per-experiment YAML, mirroring the
stage-2 config style. A separate `configs/e2e/` folder with e2e YAMLs. Stage-2
stays untouched.

Keep exactly **three** decoder experiments:

| # | Name | weighting | projector | cls_surrogate | sigreg | layers |
|---|------|-----------|-----------|---------------|--------|--------|
| 1 | Random Drop + MLP + SigReg | dropmean | bn | false | on (λ=0.02, global, N-scaled) | 1..23 |
| 2 | Random Drop + Decoder      | dropmean | none | false | off | 1..23 |
| 3 | raev2 K=23                 | mean | none | **true** | off | 1..23 |

Experiment 1 MLP norm defaults to **BatchNorm** (matches the running
`dropmean_bn` job and the LeWM recipe); switching to LayerNorm is a one-line
config change (`projector: ln`).

## Directory Layout

```
configs/stage1/decoder/              # NEW subdir — user's MLS-decoder retraining configs
  dropmean-bn-sigreg-k23.yaml        # experiment 1
  dropmean-plain-k23.yaml            # experiment 2
  raev2-k23.yaml                     # experiment 3
configs/e2e/                         # NEW folder
  e2e-nodrop.yaml
  e2e-drop.yaml
```

Kept separate from the official `configs/stage1/training/` (which targets
`stage1.RAE`) to avoid collision.

## Config Schema

```yaml
# configs/stage1/decoder/dropmean-bn-sigreg-k23.yaml
combine:
  target: stage1.combine.MLSCombine
  params:
    layers: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    weighting: dropmean        # mean | dropmean | softgate
    p_drop: 0.3                # used when weighting in {dropmean, softgate}
    projector: bn              # none | ln | bn
    cls_surrogate: false       # add L23 token-mean (raev2)
    dim: 1024
    out_dim: 1024
    mult: 4                    # projector hidden = dim*mult
decoder:
  config_path: configs/decoder/ViTXL
  latent_dim: 1024
loss:
  lpips_w: 1.0
  sigreg:                      # null disables SIGReg entirely
    weight: 0.02
    distributed: true
    scale_by_n: true
  gan:
    disc_weight: 0.75
    disc_start: 1              # epoch GAN turns on
    disc_ckpt: pretrained_models/encoders/dino/dino_vit_small_patch8_224.pth
data:
  data_dir: /datasets/imagenet-256-full
  image_size: 256
  val_npz: data_eval/imagenet-256-val.npz
  val_n: 1000
training:
  epochs: 10
  batch_size: 32              # per GPU
  lr: 8.0e-4
  warmup_epochs: 2
  ema_decay: 0.9995
  clip_grad: 1.0
  precision: bf16
  ckpt_every: 1
  log_every: 50
  seed: 42
  out_dir: output_full/dropmean_bn_sigreg_k23
probe:
  loo_solo: final              # off | final | every
  val_image: assets/samples/sample_1.png   # the 5 demo imgs (progress signal)
wandb:
  enabled: true
  project: raev3-full
  entity: uscgvl
  name: decoder-dropmean-bn-sigreg-k23
```

## Components

### `src/stage1/combine.py` (NEW)

One `MLSCombine(nn.Module)` collapsing all variants. Interface:

```python
forward(layer_tokens: list[Tensor], idx: Optional[list[int]] = None) -> Tensor  # [B, N, out_dim]
```

- `weighting`:
  - `mean` — full equal-weight mean over all K layers.
  - `dropmean` — train: equal-weight mean over a random per-sample kept subset
    (Bernoulli `p_drop`, ≥1 kept); eval: full mean.
  - `softgate` — learnable per-layer softmax weights (+ optional `p_drop`).
    Retained for completeness; not used by the 3 kept experiments.
- `cls_surrogate` — add `layer_tokens[-1].mean(dim=1, keepdim=True)` to z0
  (raev2 combine).
- `projector`:
  - `none` — return z0 directly (0 params).
  - `ln` — `skip(z0) + fc2(GELU(LayerNorm(fc1(z0))))` pre-LN residual MLP.
  - `bn` — `skip(z0) + fc2(GELU(BatchNorm1d(fc1(z0))))` over B·N token samples
    (LeWM recipe; output is a bare Linear so SIGReg sees an unconstrained dist).
- `idx` — restricts the combine to a layer subset (renormalized), powering the
  LOO/solo probes from a single trained model.

`MLSCombine.parameters()` is empty for `none`/`mean`, so the trainer must guard
the DDP wrap (`DDP` errors on a param-free module — call it directly instead).

### `src/train_decoder.py` (NEW, ~400 lines)

Single config-driven trainer capturing the user's recipe:
- Loads config via `OmegaConf.merge(structured(DecoderConfig), load(path))`.
- Frozen DINOv3 encoder → `instantiate_from_config(config.combine)` → ViT-XL
  decoder (from scratch).
- Loss = L1 + `lpips_w`·LPIPS + (GAN from `disc_start`, adaptive-weighted) +
  (`sigreg.weight`·SIGReg if `loss.sigreg` is not null).
- EMA of combine+decoder (buffers copied, not EMA'd — for BN running stats).
- Per-epoch val: 5 demo imgs (`val/psnr_demo`) + random `val_n` val images
  (`val/psnr`, `val/ssim`, paper protocol).
- LOO/solo probes per `probe.loo_solo` (final epoch by default).
- Auto-resume from `<out_dir>/ckpt_latest.pt` (incl. scheduler state).

### `src/configs/stage1_decoder.py` (NEW)

OmegaConf `@dataclass` schema (`DecoderConfig` with nested
`CombineConfig`/`LossConfig`/`DataConfig`/`TrainingConfig`/`ProbeConfig`/
`WandbConfig`), mirroring `src/configs/stage2.py`.

### `run_decoder.sh <config.yaml>` (NEW)

Thin launcher: resolve NGPU / CUDA_VISIBLE_DEVICES / wandb key, pick a free
rendezvous port, `torchrun src/train_decoder.py --config <yaml>`.

### E2E

`src/train_e2e_sigreg_dit.py` gains a `--config` loader (its CLI flags stay as
overrides). `configs/e2e/e2e-{nodrop,drop}.yaml` capture the two variants
(layer-drop, init, lr, epochs, sigreg λ, w_pix, etc.). The e2e trainer already
exists and works; this is a config wrapper, not a rewrite.

## Data Flow

```
imgs[0,1] → frozen DINOv3 (layers) → K×tokens
          → MLSCombine (weighting/dropout/surrogate/projector) → z[B,N,C]
          → SIGReg(z) [if enabled]   (logged + optional grad to projector)
          → ViT-XL decoder → x_rec → L1 + LPIPS + GAN vs imgs
```

LOO/solo: re-run the frozen combine+decoder with `idx` = each subset, no
retraining.

## Validation / Migration

1. Build trainer + combine + configs (additive; running scripts untouched).
2. **Reproduction check**: run `src/train_decoder.py` with a config replicating
   `dropmean_bn` for ~100 steps; confirm loss / PSNR / SIGReg / z-stats
   trajectory matches the legacy `train_decoder_mls_dropmean_bn_sigreg.py`
   within noise (same seed → near-identical).
3. After the 3 currently-running experiments finish **and** the reproduction
   check passes → delete the 9 legacy `train_decoder_mls*.py` and their
   `run_train_decoder_*.sh` / `run_all*decoder*.sh`.

## Out of Scope (YAGNI)

- Stage-2 changes (already config-driven).
- KL-regularized variant (`train_decoder_mls_kl.py`) — not in the 3 kept
  experiments; deleted in migration.
- softgate experiments — `MLSCombine` keeps the `softgate` weighting option but
  no config ships for it.

## Open Questions

- Experiment 1 MLP norm: defaulting to **BN**; flip to `ln` if the LeWM-style
  BN projector underperforms.
