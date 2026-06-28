#!/usr/bin/env bash
# SINGLE-NODE (8xH100) Pluto entry for stage-1 DECODER training (train_decoder.py).
# train_decoder.py is config-driven (only --config); batch/lr/epochs/out_dir/init_from/
# wandb all live in the YAML. It auto-resumes from <out_dir>/ckpt_latest.pt; with no ckpt
# and no init_from it trains from scratch.
#
# Pluto job: 1 replica, 8 GPUs. Scripts field:
#   export WANDB_KEY=...
#   bash /sensei-fs-3/users/hongyangd/RAEv3/RAEv2/run_pluto_decoder.sh drop0-scratch
set -uo pipefail

for base in /sensei-fs-3 /mnt/remotes/sensei-fs-3; do
  if [ -d "$base/users/hongyangd/RAEv3/RAEv2" ]; then REPO="$base/users/hongyangd/RAEv3/RAEv2"; ROOT="$base/users/hongyangd"; break; fi
done
: "${REPO:?could not find RAEv3/RAEv2 on the sensei mount}"
cd "$REPO"

mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/${JOB_NAME:-decoder}-node${RANK:-0}.log") 2>&1
echo "================ $(date '+%F %T')  host=$(hostname)  rank=${RANK:-0}  ================"

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
[ -x "$PY" ] || { echo "FATAL: portable env not found at $ROOT/rae_env"; exit 1; }

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
[ -n "${WANDB_KEY:-}" ] && export WANDB_API_KEY="${WANDB_KEY}"
export WANDB_ENTITY="${WANDB_ENTITY:-uscgvl}"
export WANDB_PROJECT="${WANDB_PROJECT:-omnirae}"

case "${1:-}" in
  drop0-scratch) CFG=configs/stage1/decoder/ourpipe-drop0-k23-16ep.yaml ;;
  ft-plain)      CFG=configs/stage1/decoder/ft-xcong-plain-k23-nodrop.yaml ;;
  *) echo "usage: bash run_pluto_decoder.sh <drop0-scratch|ft-plain>"; exit 1 ;;
esac

NGPU="$(${PY} -c 'import torch;print(torch.cuda.device_count())')"
echo "### $(date '+%F %T')  decoder  py=$PY  cfg=$CFG  ngpu=$NGPU  wandb=$([ -n "${WANDB_API_KEY:-}" ] && echo on || echo off)"

exec "$TR" --standalone --nproc_per_node="${NGPU:-8}" \
  src/train_decoder.py --config "$CFG"
