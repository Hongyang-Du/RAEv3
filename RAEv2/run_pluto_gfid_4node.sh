#!/usr/bin/env bash
# 4-NODE gFID + IS eval. Each node runs ONE condition independently on its 8 local
# GPUs (NOT cross-node DDP) -- node RANK r runs CONDS[r]. All 4 conditions finish
# in parallel (~45 min). offline_eval splits that condition's 50k across the node's
# 8 GPUs, gathers, and computes FID + IS (torch-fidelity) vs our evanarlian val npz.
# Guidance = official RAEv2 imagenet (IG only): noguid=ig1.0, ig178=ig1.78 t_min0.10.
#
# Pluto job: 4 replicas, 8 GPUs/replica. Scripts field (one line, no args):
#   bash /sensei-fs-3/users/hongyangd/RAEv3/RAEv2/run_pluto_gfid_4node.sh
set -uo pipefail

for base in /sensei-fs-3 /mnt/remotes/sensei-fs-3; do
  if [ -d "$base/users/hongyangd/RAEv3/RAEv2" ]; then REPO="$base/users/hongyangd/RAEv3/RAEv2"; ROOT="$base/users/hongyangd"; break; fi
done
: "${REPO:?could not find RAEv3/RAEv2 on the sensei mount}"
cd "$REPO"

mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/${JOB_NAME:-gfid4node}-node${RANK:-0}.log") 2>&1
echo "================ $(date '+%F %T')  host=$(hostname)  rank=${RANK:-0}  ================"

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
[ -x "$PY" ] || { echo "FATAL: portable env not found at $ROOT/rae_env"; exit 1; }

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# node RANK -> condition (each node independent; 4 nodes cover the 2x2)
CONDS=(h1plain-noguid h1plain-ig178 encoder-noguid encoder-ig178)
R="${RANK:-0}"
cond="${CONDS[$R]:-}"
[ -z "$cond" ] && { echo "node $R: no condition (need exactly 4 nodes for the 4 conditions); idle."; exit 0; }

NGPU="${NUM_OF_GPUS:-$(${PY} -c 'import torch;print(torch.cuda.device_count())')}"
export EXPERIMENT_NAME="omnirae-gfid-${cond}"
CFG="configs/stage2/sampling/omnirae-eval-${cond}.yaml"
[ -f "$CFG" ] || { echo "FATAL: no config $CFG"; exit 1; }
echo "### $(date '+%F %T')  node $R -> ${cond}  ngpu=${NGPU}  cfg=${CFG}  exp=${EXPERIMENT_NAME}"

# --standalone: each node rendezvous on its OWN localhost (independent 8-GPU group),
# so the 4 nodes do NOT join one DDP group. Per-node rdzv-id keeps them isolated.
"$TR" --standalone --nproc_per_node="${NGPU:-8}" --rdzv-id="gfid-${cond}" \
  src/offline_eval.py --config "$CFG"

CSV="results/stage2/eval/${EXPERIMENT_NAME}_ema.csv"
echo "### $(date '+%F %T')  node $R done ${cond}"
if [ -f "$CSV" ]; then echo "### RESULT ($CSV):"; cat "$CSV"; echo; else echo "### WARN: no CSV at $CSV"; fi
echo "Done node $R."
