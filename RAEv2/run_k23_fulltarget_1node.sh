#!/usr/bin/env bash
# ============================================================
#  SINGLE-NODE (8xA100) DiT-B decoupled FULL TARGET, k23 plain.
#  Parameterized by p-tag. Computes latent_stats.pt if missing, then trains.
#  Usage:  nohup bash run_k23_fulltarget_1node.sh p03 > LOG 2>&1 &
#  Valid tags: p00 p005 p01 p03 p05 p07 p09
# ============================================================
set -uo pipefail
TAG="${1:?usage: bash run_k23_fulltarget_1node.sh <p-tag e.g. p03>}"
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
CFG="configs/stage2/training/imagenet-dinov3l-omni-randomdrop-fulltarget-plain-k23-${TAG}-ditb.yaml"
[ -f "$CFG" ] || { echo "FATAL: config missing: $CFG"; exit 1; }

# Experiment name = the dir that the config's normalization_stat_path lives in
# (single source of truth -> no drift between config and ckpt dir).
STATS=$(grep -E '^\s*normalization_stat_path:' "$CFG" | awk '{print $2}')
NAME=$(basename "$(dirname "$STATS")")
export EXPERIMENT_NAME="$NAME"
DATA=/mnt/localssd/imagenet-256

mkdir -p "$ROOT/ckpt/$NAME" "$ROOT/logs"
echo "================ $(date '+%F %T') host=$(hostname) $NAME tag=$TAG nproc=$NPROC ================"

# preflight
[ -d "$DATA/imagenet-latents-images" ] || { echo "FATAL: imagenet-256 not staged at $DATA"; exit 1; }
# free-GPU sanity: warn loudly if the GPUs are already busy (another run will OOM)
BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{c++} END{print c+0}')
[ "$BUSY" -gt 0 ] && echo "### WARNING: ${BUSY} GPU(s) already have >2GB used -- concurrent runs on the same GPUs will OOM."

# 1. latent stats
if [ ! -f "$STATS" ]; then
  echo "### $(date '+%F %T') computing latent_stats -> $STATS"
  "$TR" --standalone --nproc_per_node="$NPROC" scripts/stage1/compute_latent_stats.py \
    --config "$CFG" --data-dir "$DATA" --output-path "$STATS" --num-samples 250000 \
    || { echo "FATAL: compute_latent_stats failed"; exit 1; }
else
  echo "### latent_stats already present: $STATS"
fi
[ -f "$STATS" ] || { echo "FATAL: stats missing after compute"; exit 1; }

# 2. train
echo "### $(date '+%F %T') launching training: $NAME"
exec "$TR" --standalone --nproc_per_node="$NPROC" \
  src/train.py \
  --config "$CFG" \
  --results-dir "$ROOT/ckpt" \
  --precision bf16
