#!/usr/bin/env bash
# ============================================================
#  SINGLE-NODE (8xA100) run: DiT-B decoupled FULL TARGET, k23 plain, p_drop=0.0
#  ("no dropout" k23 baseline). Computes latent_stats.pt if missing, then trains.
#  Launch:  nohup bash run_k23_fulltarget_p00_1node.sh &
# ============================================================
set -uo pipefail
ROOT=/sensei-fs-3/users/hongyangd
REPO=$ROOT/RAEv3_oldnorm/RAEv2
cd "$REPO"

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
[ -x "$PY" ] || { echo "FATAL: env missing at $ROOT/rae_env"; exit 1; }

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CKPT_KEEP_RECENT="${CKPT_KEEP_RECENT:-2}"
export CKPT_KEEP_EVERY="${CKPT_KEEP_EVERY:-10}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-500}"

NPROC="${NUM_OF_GPUS:-8}"
CFG="configs/stage2/training/imagenet-dinov3l-omni-randomdrop-fulltarget-plain-k23-p00-ditb.yaml"
NAME="dit-b-omni-randomdrop-fulltarget-plain-k23-p0.0"
export EXPERIMENT_NAME="$NAME"
DATA=/mnt/localssd/imagenet-256
STATS="$ROOT/ckpt/$NAME/latent_stats.pt"

mkdir -p "$ROOT/ckpt/$NAME" "$ROOT/logs"
echo "================ $(date '+%F %T') host=$(hostname) $NAME nproc=$NPROC ================"

# preflight: data + config
[ -d "$DATA/imagenet-latents-images" ] || { echo "FATAL: imagenet-256 not staged at $DATA"; exit 1; }
[ -f "$CFG" ] || { echo "FATAL: config missing: $CFG"; exit 1; }

# 1. latent stats (drop=true + p_drop=0.0 -> stats over the full-mean latent, self-consistent)
if [ ! -f "$STATS" ]; then
  echo "### $(date '+%F %T') computing latent_stats -> $STATS"
  "$TR" --standalone --nproc_per_node="$NPROC" scripts/stage1/compute_latent_stats.py \
    --config "$CFG" --data-dir "$DATA" --output-path "$STATS" --num-samples 250000 \
    || { echo "FATAL: compute_latent_stats failed"; exit 1; }
else
  echo "### latent_stats already present: $STATS"
fi
[ -f "$STATS" ] || { echo "FATAL: stats missing after compute"; exit 1; }

# 2. train (bf16, gbs=2048 over 8 GPUs -> 256/GPU micro-batch)
echo "### $(date '+%F %T') launching training: $NAME"
exec "$TR" --standalone --nproc_per_node="$NPROC" \
  src/train.py \
  --config "$CFG" \
  --results-dir "$ROOT/ckpt" \
  --precision bf16
