# RAEv2 Training Handoff — Decoder + DiT (plain & sigreg)

Full pipeline for training, from a clean machine, the two decoder variants and their
stage-2 DiTs:

| # | stage | epochs | config |
|---|-------|--------|--------|
| 1 | **plain** random-drop decoder        | 16 | `configs/stage1/decoder/random-drop-layer-mls-plain-k23.yaml` |
| 2 | **plain** DiT (on the decoder latent)| 80 | `configs/stage2/training/imagenet-dinov3l-k23-drop-plain.yaml` |
| 3 | **sigreg** random-drop decoder (+projector+SIGReg) | 16 | `configs/stage1/decoder/random-drop-layer-mls-mlp-sigreg-k23.yaml` |
| 4 | **sigreg** DiT (on the projected/SIGReg latent)    | 80 | `configs/stage2/training/imagenet-dinov3l-k23-drop-sigreg.yaml` |

The only difference between plain and sigreg is the stage-1 `combine`: plain = equal-weight
mean of the random-dropped layers; sigreg = same mean → a `bn`-MLP projector → SIGReg
gaussianized latent. The DiT generates whatever the combine outputs; the decoder reconstructs.

---

## 0. Hardware
8× A100/H100 (80 GB) assumed below. Fewer GPUs: scale `--nproc_per_node` and (if OOM) add
`grad_accum_steps` in the config to keep the global batch.

## 1. Docker + environment

```bash
# base image used in-house
docker run -d --name rae_train --gpus all --ipc=host \
  -v /datasets:/datasets \
  -v $HOME/.ssh:/root/.ssh \
  -v /path/to/RAEv3:/workspace/RAEv3 \
  nvcr.io/nvidia/pytorch:25.04-py3 sleep infinity

docker exec -it rae_train bash
cd /workspace/RAEv3/RAEv2

# build the `rae` conda env (miniconda + torch 2.10.0+cu128 + deps). ~10-15 min, idempotent.
bash setup_rae_env.sh
export PY=/opt/conda/envs/rae/bin/python
export TR=/opt/conda/envs/rae/bin/torchrun

# gmuon optimizer (used by the DiT)
${PY} -m pip install -e third_party/gmuon --no-deps

# wandb (optional; drop --wandb below if unused)
export WANDB_API_KEY=<your_key>
export WANDB_ENTITY=<your_entity> WANDB_PROJECT=rae-stage2
```

## 2. Download ImageNet-256 (HuggingFace → Arrow)

```bash
# source: evanarlian/imagenet_1k_resized_256 ; saved as memory-mapped Arrow shards.
# --num-samples controls size: full train ≈ 1,281,167. Output dir MUST match the
# data_dir in the configs (/datasets/imagenet-256-full).
${PY} scripts/data/download_imagenet256.py \
  --hf-dataset evanarlian/imagenet_1k_resized_256 \
  --out-dir /datasets/imagenet-256-full \
  --num-samples 1281167 --num-shards 256
# -> creates /datasets/imagenet-256-full/imagenet-latents-images/data-*.arrow
```
(Reconstruction eval needs `data_eval/imagenet-256-val.npz`, a 50k-image uint8 NPZ — ship it
with the repo or regenerate from the val split.)

## 3. Training pipeline

Each DiT needs three steps: train the decoder → compute its latent stats → train the DiT.
All paths below are already wired in the configs; just run in order.

### Plain (steps 1–2)
```bash
# 1) plain decoder, 16 epochs  -> output_full/decoder_random_drop_layer_mls_plain_k23/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_ALLOC_CONF=expandable_segments:True \
  ${TR} --nproc_per_node=8 --master-port=29501 src/train_decoder.py \
  --config configs/stage1/decoder/random-drop-layer-mls-plain-k23.yaml

# 2a) latent stats for the plain DiT  -> .../latent_stats.pt (path the DiT config expects)
CUDA_VISIBLE_DEVICES=0 ${TR} --nproc_per_node=1 src/compute_h1_stats.py \
  --config configs/stage2/training/imagenet-dinov3l-k23-drop-plain.yaml \
  --num-samples 50000 \
  --out output_full/decoder_random_drop_layer_mls_plain_k23/latent_stats.pt

# 2b) plain DiT, 80 epochs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 EXPERIMENT_NAME=dit-k23-drop-plain-80ep \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  ${TR} --nproc_per_node=8 --master-port=29502 src/train.py \
  --config configs/stage2/training/imagenet-dinov3l-k23-drop-plain.yaml \
  --results-dir ckpts_full/stage2 --precision bf16 --wandb
```

### Sigreg (steps 3–4)
```bash
# 3) sigreg decoder (projector + SIGReg), 16 epochs
#    -> output_full/decoder_random_drop_layer_mls_mlp_sigreg_k23/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_ALLOC_CONF=expandable_segments:True \
  ${TR} --nproc_per_node=8 --master-port=29503 src/train_decoder.py \
  --config configs/stage1/decoder/random-drop-layer-mls-mlp-sigreg-k23.yaml

# 4a) latent stats for the sigreg DiT
CUDA_VISIBLE_DEVICES=0 ${TR} --nproc_per_node=1 src/compute_h1_stats.py \
  --config configs/stage2/training/imagenet-dinov3l-k23-drop-sigreg.yaml \
  --num-samples 50000 \
  --out output_full/decoder_random_drop_layer_mls_mlp_sigreg_k23/latent_stats.pt

# 4b) sigreg DiT, 80 epochs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 EXPERIMENT_NAME=dit-k23-drop-sigreg-80ep \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  ${TR} --nproc_per_node=8 --master-port=29504 src/train.py \
  --config configs/stage2/training/imagenet-dinov3l-k23-drop-sigreg.yaml \
  --results-dir ckpts_full/stage2 --precision bf16 --wandb
```

## 4. Notes
- **Decoders**: 16 epochs, GAN on from epoch 1, batch 256 (32/GPU × 8). Output = `ckpt_latest.pt`
  (full) — for distribution a stripped inference-only ckpt (ema_dec + ema_combine) suffices.
- **DiTs**: 80 epochs, DDT-XL, gmuon (lr 2e-4, linear warmup 25 / decay-end 50), global batch
  1024, grad_accum 1 → 128/GPU on 8 GPUs. If OOM, set `grad_accum_steps: 2` in the DiT config.
- **Pre-trained decoders (skip steps 1/3)** are on Google Drive:
  `RAEv2.5_decoder_dropmean_nosigreg_k23.pt` / `RAEv2.5_decoder_dropmean_sigreg_k23.pt`
  (+ their `*_latent_stats*.pt`). Place them at the paths the DiT configs expect.
