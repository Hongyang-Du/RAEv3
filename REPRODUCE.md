# REPRODUCE — RAEv3 MLS decoder (`nogate_sigreg`) + DiT (fresh-machine run book)

This document rebuilds, **from an empty machine**, the single canonical two-stage pipeline of this repo:

> **Stage 1 — `mls_nogate_sigreg`:** DINOv3-L (frozen, K=7 layers) → fixed-mean MLS combine →
> learnable SIGReg projector → ViT-XL decoder (trained from scratch).
> **Target:** held-out **val PSNR ≈ 26.02 dB** at epoch 5.
>
> **Stage 2 — DiT `nogate`:** DDT-XL flow-matching generator on the frozen stage-1 latents.
> **Target:** **FID(5k, no guidance) ≈ 16.9** at epoch 10.

The two stages are a chain: stage 2 consumes the stage-1 checkpoint. Both are trained **from
scratch** here (no checkpoint of ours is required — only the public DINOv3-L encoder + ImageNet).

Out of scope (kept in the repo for reference): the `raev2mls` / `dropmean` / `softgate` / `l11`
variants, the E2E joint-training path (`train_e2e_sigreg_dit.py`, needs the empty `third_party/le-wm`),
and all FID/linear-probe eval tooling.

All code lives in the **`RAEv2/`** subdirectory; run everything from there.

---

## 0. What you are reproducing (at a glance)

| Item | Value |
|---|---|
| Stage-1 entry | `RAEv2/src/train_decoder_mls_nogate_sigreg.py` via `run_train_decoder_mls_nogate_sigreg.sh` |
| Stage-2 entry | `RAEv2/src/train.py` via `run_train_dit.sh nogate` (auto-computes latent stats first) |
| Encoder | DINOv3-L (`dinov3_vitl16`), **frozen**, layers 11,13,15,17,19,21,23 |
| Combine | fixed unweighted mean over the 7 layers (no learnable gate) |
| Projector | learnable per-token residual MLP, shaped by SIGReg (1024 projections, multi-freq Epps-Pulley) |
| Decoder | ViT-XL, trained from scratch (the only trainable module in stage 1) |
| Stage-1 loss | L1 + LPIPS + GAN + `sigreg_w·SIGReg` (`sigreg_w=1`, `lpips_w=1.0`, `disc_weight=0.75`) |
| Stage-2 model | DDT-XL (28×1440 + 2×2048 head, internal guidance @ depth 8), x-pred flow matching, logit-normal t, shift 8 |
| Dataset | ImageNet-1k resized to 256, HF arrow format (`<data_dir>/imagenet-latents-images/`) |
| Distribution | `torchrun`, 8 GPUs, bf16; stage-1 batch 32/GPU, stage-2 global batch 1024 |
| Schedule | stage-1: 5 epochs, lr 8e-4 (AdamW) · stage-2: 10 epochs, gmuon lr 2e-4 |
| Outputs | stage-1 `RAEv2/output_full/train_decoder_mls_nogate_sigreg/ckpt_latest.pt`; stage-2 `RAEv2/ckpts_full/stage2/dit-nogate-k7/` |

---

## 1. Hardware / prerequisites

- **GPUs:** 8× NVIDIA A100/H100 80GB (both scripts hardcode `NGPU=8`).
- **uv** (the project's package manager): `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **GitHub SSH access to the private `nanovisionx` org** — required by `uv sync` (see §3). On a Pluto
  node, load the key you use for `github.com/nanovisionx`.
- **Meta DINOv3 license access** to download the encoder weights (see §4a).

### Configurable roots (set once)

The repo's scripts contain two **stale hardcoded paths** that you must redirect on a new machine:
`CONDA_ENV=/opt/conda/envs/rae` (we use the uv `.venv` instead) and `DATA=/datasets/imagenet-256-full`.
Set these and use them consistently:

```bash
export RAE_ROOT=/path/to/your/workspace/RAEv3/RAEv2   # the RAEv2 package dir
export DATA_DIR=/datasets/imagenet-256-full            # what the scripts expect (see §5 for the symlink)
export DINOV3_CKPT_DIR=$RAE_ROOT/pretrained_models/encoders/dinov3
```

---

## 2. Get the code

```bash
git clone <your RAEv3 remote> RAEv3       # the repo containing RAEv2/, STATUS.md, third_party/
cd RAEv3/RAEv2
export RAE_ROOT=$PWD
```

---

## 3. Build the environment (uv, Torch 2.10.0 + cu128)

The real environment is the project's uv `.venv` (Python 3.10, **torch 2.10.0+cu128**), declared in
`pyproject.toml`. **Ignore** the `/opt/conda/envs/rae` paths inside the `run_*.sh` scripts — they are
container artifacts of the original DGX session.

```bash
cd "$RAE_ROOT"
uv sync          # creates .venv from pyproject.toml + uv.lock; torch comes from the cu128 index
```

`uv sync` pulls four **private** git deps from `github.com/nanovisionx` plus OpenAI CLIP
(`pyproject.toml [tool.uv.sources]`):
- `gram-newton-schulz` (**gmuon optimizer — REQUIRED for stage-2 DiT**, `gmuon lr 2e-4`)
- `dpg-evaluator`, `geneval-evaluator`, `t2v-metrics`, `fd-evaluator` (eval only — FID is **disabled**
  in the training configs, so these are not needed to train)
- `clip` (`github.com/openai/CLIP`, public)

> **If you lack `nanovisionx` SSH access:** stage-1 decoder training does **not** import gmuon and runs
> fine without it; you can comment the four `nanovisionx` lines out of `[tool.uv.sources]` +
> `dependencies` and `uv sync` the rest. **Stage-2 DiT requires gmuon** — you must have access (or a
> local copy of `nanovisionx/gmuon`) for it.

Run commands either with `uv run <cmd>` or by activating `source $RAE_ROOT/.venv/bin/activate`.

Sanity check:
```bash
uv run python -c "import torch; print('torch', torch.__version__, torch.version.cuda, 'gpus', torch.cuda.device_count())"
# expect: torch 2.10.0+cu128 12.8 ...
```

---

## 4. Download weights

### 4a. Encoder — DINOv3-L (gated, Meta)

DINOv3 weights are gated. Request access (HF `facebook/dinov3-vitl16-pretrain-lvd1689m` or the
`facebookresearch/dinov3` repo), then place the file where the loader expects it:

```bash
mkdir -p "$DINOV3_CKPT_DIR"
# obtain: dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth  (~1.2 GB)  -> $DINOV3_CKPT_DIR/
ls -la "$DINOV3_CKPT_DIR/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
```

The loader (`src/encoders/models/dinov3_loader.py:34-77`) resolves the file from `DINOV3_CKPT_DIR`
(default `RAEv2/pretrained_models/encoders/dinov3/`), validates the `8aa4cbdd` hash, and pulls the
model definition from `torch.hub` ref `facebookresearch/dinov3:94a96ac…`. Two env overrides:
- `DINOV3_CKPT_DIR` — directory holding the `.pth` (set above).
- `DINOV3_REPO_DIR` — path to a local clone of the dinov3 repo (with `hubconf.py`) to avoid the
  `torch.hub` GitHub fetch on an offline node. Optional but recommended for offline Pluto nodes.

> Only `dinov3_vitl16` is needed for this pipeline. (The other sizes seen in the original cache —
> vits/vitb/vith/vit7b — are for other experiments and are not used here.)

### 4b. LPIPS

`lpips==0.1.4` downloads its AlexNet/VGG weights from the network on first use; if the node is
offline, pre-warm them on a connected machine and copy `~/.cache/torch/hub/checkpoints/`.

---

## 5. Data — ImageNet-1k @ 256 (public)

The arrow store was built from the **public** HF dataset
[`evanarlian/imagenet_1k_resized_256`](https://huggingface.co/datasets/evanarlian/imagenet_1k_resized_256)
(commit `8de107e…`, ~25 GB), saved into `<data_dir>/imagenet-latents-images/` as 64 arrow shards.
The loaders look for exactly that subdir (`…:99` `arrow_dir = os.path.join(data_dir, "imagenet-latents-images")`).

```bash
# Build the arrow store the loader expects (one-time, ~25 GB):
export HF_HOME=/path/to/hf_cache
uv run python - <<'PY'
import os
from datasets import load_dataset
data_dir = os.environ["DATA_DIR"]                       # /datasets/imagenet-256-full
out = os.path.join(data_dir, "imagenet-latents-images")
ds = load_dataset("evanarlian/imagenet_1k_resized_256", split="train")   # public
ds.save_to_disk(out)
print("wrote", out, ds)
PY
```

**Path note:** the scripts and the stage-2 config (`configs/stage2/training/imagenet-dinov3l-k7-nogate-sigreg.yaml:56`)
both hardcode `DATA=/datasets/imagenet-256-full`. Either build directly into that path (above) or, if
your data lives elsewhere, create the symlink the repo uses:
```bash
sudo mkdir -p /datasets && sudo ln -s "$REAL_DATA_PARENT/imagenet-256-full" /datasets/imagenet-256-full
# (the repo also keeps RAEv2/data/imagenet-256 -> the same target for convenience)
```

> **Shortcut if a sensei-fs cache is mounted:** the prebuilt arrow already exists at
> `/sensei-fs-3/users/yunfeix/datasets_backup/imagenet-256-full/imagenet-latents-images/` — symlink
> `/datasets/imagenet-256-full` to `…/datasets_backup/imagenet-256-full` and skip the build.

---

## 6. Experiment tracking

Both scripts read the wandb key from `~/.netrc` and log to project `raev3-full`, entity `uscgvl`:
```bash
cat >> ~/.netrc <<EOF
machine api.wandb.ai
  login user
  password <your-wandb-key>
EOF
chmod 600 ~/.netrc
```
Change `WANDB_PROJECT`/`WANDB_ENTITY` at the top of the run scripts to your own, or pass `WANDB=false`
to `run_train_decoder_mls_nogate_sigreg.sh` to disable.

---

## 7. Launch — stage 1 (decoder)

Point the script at the uv venv (instead of the stale conda path) and run:

```bash
cd "$RAE_ROOT"
# one-time: redirect the env + data lines of the script to this machine
sed -i "s#^CONDA_ENV=.*#CONDA_ENV=$RAE_ROOT/.venv#" run_train_decoder_mls_nogate_sigreg.sh
sed -i "s#^DATA=.*#DATA=$DATA_DIR#"                  run_train_decoder_mls_nogate_sigreg.sh

bash run_train_decoder_mls_nogate_sigreg.sh
```
This runs (8 GPU, batch 32/GPU, 5 epochs, bf16, lr 8e-4, layers `11 13 15 17 19 21 23`, `sigreg_w=1`)
and writes `output_full/train_decoder_mls_nogate_sigreg/ckpt_latest.pt` (+ `ckpt_epXX.pt` each epoch).

**Verify:** val PSNR climbs to **≈ 26.02 dB** by epoch 5 (`output_full/train_decoder_mls_nogate_sigreg/train.log`;
compare `plot_val_psnr.py`).

---

## 8. Launch — stage 2 (DiT on the frozen stage-1 latents)

```bash
cd "$RAE_ROOT"
sed -i "s#^CONDA_ENV=.*#CONDA_ENV=$RAE_ROOT/.venv#" run_train_dit.sh
sed -i "s#^DATA=.*#DATA=$DATA_DIR#"                  run_train_dit.sh

bash run_train_dit.sh nogate
```
This (1) computes `output_full/train_decoder_mls_nogate_sigreg/latent_stats.pt` (Welford mean/var over
250k encoded samples) if missing, then (2) trains DDT-XL (`src/train.py` +
`configs/stage2/training/imagenet-dinov3l-k7-nogate-sigreg.yaml`, global batch 1024, gmuon lr 2e-4,
10 epochs) into `ckpts_full/stage2/dit-nogate-k7/`.

It refuses to start unless `output_full/train_decoder_mls_nogate_sigreg/ckpt_latest.pt` exists, and
warns if a stage-1 job is still writing it.

**Verify:** FID(5k, no guidance) ≈ **16.9** at epoch 10. Note the configs ship with FID eval
**disabled** (no val split / reference npz on the box); the in-training **denoise-probe PSNR**
(`stage2/utils.denoise_probe`, `plot_dit_progress.py`) is the per-epoch progress signal.

---

## 9. Smoke test (confirm the pipeline works before the full runs)

Stage-1 smoke — real code path (DINOv3 load + arrow loader + projector + decoder + GAN + SIGReg +
optimizer steps), tiny and short, on 1 GPU. Edit a copy so it exits after a few steps:

```bash
cd "$RAE_ROOT" && source .venv/bin/activate
export DINOV3_CKPT_DIR=$RAE_ROOT/pretrained_models/encoders/dinov3
PYTORCH_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=1 \
  src/train_decoder_mls_nogate_sigreg.py \
  --data "$DATA_DIR" --out-dir output_full/_smoke_nogate \
  --epochs 1 --batch-size 4 --precision bf16 --lr 8e-4 \
  --layers 11 13 15 17 19 21 23 --sigreg-w 1 --lpips-w 1.0 \
  --disc-weight 0.75 --disc-start 1 \
  --ckpt-every 1 --val-every 5 --log-every 1 --val-image assets/samples/sample_1.png
```
Success = DINOv3-L loads, the arrow dataset iterates, a finite L1/LPIPS/GAN/SIGReg loss prints, and
`output_full/_smoke_nogate/` gets a checkpoint. Kill after a few steps. (For a stage-2 smoke, ensure a
stage-1 `ckpt_latest.pt` exists — copy the smoke ckpt — then run `run_train_dit.sh nogate` and Ctrl-C
once it starts stepping.)

---

## 10. Troubleshooting

- **`uv sync` fails on `nanovisionx` git deps:** missing SSH access. For stage-1 only, drop those four
  lines (see §3). For stage-2, you must have access to `nanovisionx/gmuon`.
- **DINOv3 hash/`weights file not found`:** the `.pth` is missing or misnamed — it must be exactly
  `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` under `DINOV3_CKPT_DIR` (§4a).
- **`torch.hub` GitHub fetch hangs (offline node):** set `DINOV3_REPO_DIR` to a local dinov3 clone.
- **`imagenet-latents-images not found` / falls back to ImageFolder:** `data_dir` must contain the
  arrow subdir (§5); check the `DATA=`/symlink redirect.
- **Wrong env picked up:** the `run_*.sh` still point `TORCHRUN`/`PYTHON` at `$CONDA_ENV/bin`; make
  sure the `sed` redirect to `$RAE_ROOT/.venv` (§7-8) actually applied.

---

## 11. Provenance

| Fact | Source |
|---|---|
| Canonical = `mls_nogate_sigreg` (PSNR 26.02) → DiT nogate (FID 16.9) | `RAEv3/STATUS.md` (experiment table + stage-2 results) |
| Stage-1 args/recipe | `RAEv2/run_train_decoder_mls_nogate_sigreg.sh` |
| Stage-2 pipeline (stats → DDT-XL) | `RAEv2/run_train_dit.sh`, `configs/stage2/training/imagenet-dinov3l-k7-nogate-sigreg.yaml` |
| Env = uv `.venv`, torch 2.10.0+cu128, private nanovisionx deps | `RAEv2/pyproject.toml` (`[tool.uv.sources]`, `[[tool.uv.index]]`) |
| DINOv3-L source + path + hash | `RAEv2/src/encoders/models/dinov3_loader.py:34-77` |
| Dataset = `evanarlian/imagenet_1k_resized_256` | `datasets_backup/imagenet-256-full/imagenet-latents-images/dataset_info.json` |
| Hardcoded paths to redirect (`/opt/conda/envs/rae`, `/datasets/imagenet-256-full`) | the `run_*.sh` scripts + stage-2 yaml |
