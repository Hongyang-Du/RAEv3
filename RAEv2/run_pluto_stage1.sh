#!/usr/bin/env bash
# SINGLE-NODE (8xH100) Pluto entry for the OFFICIAL RAEv2 stage-1 decoder reproduction
# (train_stage1.py + RAE / rae.py). Official recipe; global_batch 256 -> 32/GPU on 8 GPUs.
# Auto-resumes from <results-dir>/<EXPERIMENT_NAME>.
#
# Pluto job: 1 replica, 8 GPUs. Scripts field:
#   export WANDB_KEY=...
#   bash /sensei-fs-3/users/hongyangd/RAEv3/RAEv2/run_pluto_stage1.sh repro-k23
set -uo pipefail

for base in /sensei-fs-3 /mnt/remotes/sensei-fs-3; do
  if [ -d "$base/users/hongyangd/RAEv3/RAEv2" ]; then REPO="$base/users/hongyangd/RAEv3/RAEv2"; ROOT="$base/users/hongyangd"; break; fi
done
: "${REPO:?could not find RAEv3/RAEv2 on the sensei mount}"
cd "$REPO"

mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/${JOB_NAME:-stage1-repro}-node${RANK:-0}.log") 2>&1
echo "================ $(date '+%F %T')  host=$(hostname)  rank=${RANK:-0}  ================"

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
[ -x "$PY" ] || { echo "FATAL: portable env not found at $ROOT/rae_env"; exit 1; }

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CKPT_KEEP_RECENT=4   # keep only the 4 most recent ep-*.pt (~36GB) — avoids re-filling the disk quota
export WANDB_ENTITY="${WANDB_ENTITY:-uscgvl}"
export WANDB_PROJECT="${WANDB_PROJECT:-omnirae}"
export WANDB_FRESH_RUN=1

case "${1:-}" in
  repro-k23) CFG=configs/stage1/training/repro-official-k23-16ep.yaml; export EXPERIMENT_NAME=repro-official-k23-16ep ;;
  *) echo "usage: bash run_pluto_stage1.sh <repro-k23>"; exit 1 ;;
esac

WANDB_FLAG=""; [ -n "${WANDB_KEY:-}" ] && WANDB_FLAG="--wandb"
NGPU="$(${PY} -c 'import torch;print(torch.cuda.device_count())')"
echo "### $(date '+%F %T')  ${EXPERIMENT_NAME}  py=$PY  cfg=$CFG  ngpu=$NGPU  wandb=${WANDB_FLAG:-off}"

exec "$TR" --standalone --nproc_per_node="${NGPU:-8}" src/train_stage1.py \
  --config "$CFG" \
  --results-dir "$ROOT/ckpt" \
  --precision bf16 ${WANDB_FLAG}
