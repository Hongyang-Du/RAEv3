#!/usr/bin/env bash
set -uo pipefail
ROOT=/sensei-fs-3/users/hongyangd
REPO="$ROOT/RAEv3_oldnorm/RAEv2"
CKDIR="$ROOT/ckpt/omnirae-dit-bandjepa-depthattn-k23-4node/gfid_eval"
cd "$REPO"
PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
NGPU="$(${PY} -c 'import torch;print(torch.cuda.device_count())')"

CFG="$CKDIR/cfgs/ep0024-ig178.yaml"
EVALDIR="$CKDIR/runs/ep0024-ig178"
mkdir -p "$EVALDIR"
export EXPERIMENT_NAME="bandjepa-k23-ep0024-ig178"
echo "### $(date '+%F %T')  ${EXPERIMENT_NAME}  ngpu=${NGPU}  cfg=${CFG}"
"$TR" --standalone --nproc_per_node="${NGPU:-8}" --master_port=29534 \
  src/offline_eval.py --config "$CFG" 2>&1 | tee "$EVALDIR/run.log"
CSV="$EVALDIR/${EXPERIMENT_NAME}_ema.csv"
echo "### $(date '+%F %T')  done ig178"
if [ -f "$CSV" ]; then echo "### RESULT ($CSV):"; cat "$CSV"; echo; else echo "### WARN: no CSV at $CSV"; fi
