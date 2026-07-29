#!/usr/bin/env bash
# SINGLE-NODE (1 x 8-GPU) launcher for the 16-EPOCH CLS-ON Stage-1 decoder on a FROZEN
# Stage-0 JEPA fusion. cls_surrogate=true is turned on in the decoder config (a
# non-parametric forward-additive L23 token-mean), so STAGE0_COMBINE still points at the
# ORIGINAL cls-OFF stage-0 fusion ckpt -- it strict-loads fine (the term adds no params).
# Distinct -cls out_dir (in the config) so it does NOT collide with the non-cls runs.
# Auto-resumes from <out_dir>/ckpt_latest.pt.
#
# usage:  bash run_jepa_decoder_cls_16ep_1node.sh <k23|k7>
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for base in /sensei-fs-3 /mnt/remotes/sensei-fs-3; do
  if [ -d "$base/users/hongyangd" ]; then ROOT="$base/users/hongyangd"; break; fi
done
: "${REPO:?}"; : "${ROOT:?}"; cd "$REPO"

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
[ -x "$PY" ] || { echo "FATAL: env not found at $ROOT/rae_env"; exit 1; }

case "${1:-}" in
  k23) CFG=configs/stage1/decoder/jepa-k23-stage1-decoder-cls-oldnorm.yaml
       export STAGE0_COMBINE=$ROOT/ckpt/stage0-fusion-jepa-depthattn-k23/ckpt_latest.pt ;;
  k7)  CFG=configs/stage1/decoder/jepa-k7-stage1-decoder-cls-oldnorm.yaml
       export STAGE0_COMBINE=$ROOT/ckpt/stage0-fusion-jepa-depthattn-k7/ckpt_latest.pt ;;
  *)   echo "usage: bash run_jepa_decoder_cls_16ep_1node.sh <k23|k7>"; exit 1 ;;
esac
[ -f "$STAGE0_COMBINE" ] || { echo "FATAL: STAGE0_COMBINE not found: $STAGE0_COMBINE"; exit 1; }

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CKPT_KEEP_RECENT="${CKPT_KEEP_RECENT:-3}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-500}"
# wandb off by default unless a key is exported.
[ -n "${WANDB_KEY:-}" ] && export WANDB_API_KEY="${WANDB_KEY}"
export WANDB_ENTITY="${WANDB_ENTITY:-uscgvl}"
export WANDB_PROJECT="${WANDB_PROJECT:-raev3-full}"
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline

NPROC="${NUM_OF_GPUS:-8}"
MPORT="${MASTER_PORT:-29544}"

echo "### $(date '+%F %T') 16ep CLS decoder ($1)  nproc=$NPROC  cfg=$CFG  stage0=$STAGE0_COMBINE"
exec "$TR" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  --master_port="$MPORT" \
  --rdzv_id="jepa-dec-cls-16ep-$1" \
  src/train_decoder.py \
  --config "$CFG"
