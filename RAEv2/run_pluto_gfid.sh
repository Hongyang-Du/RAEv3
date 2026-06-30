#!/usr/bin/env bash
# Distributed gFID + IS eval for the omnirae stage-2 DiTs (offline_eval.py).
# Splits the 50k samples across the job's GPUs (torchrun), then computes FID + IS
# once on rank 0 against our evanarlian held-out val npz. Guidance = official RAEv2
# imagenet setting (IG only): noguid = ig 1.0 (off), ig178 = ig 1.78 t_min 0.10.
#
# Pluto job: 1 replica, 8 GPUs (single node). For max speed submit 4 jobs in
# parallel (one per condition). Scripts field:
#   bash /sensei-fs-3/users/hongyangd/RAEv3/RAEv2/run_pluto_gfid.sh h1plain-ig178
# Conditions: h1plain-noguid | h1plain-ig178 | encoder-noguid | encoder-ig178 | all
set -uo pipefail

for base in /sensei-fs-3 /mnt/remotes/sensei-fs-3; do
  if [ -d "$base/users/hongyangd/RAEv3/RAEv2" ]; then REPO="$base/users/hongyangd/RAEv3/RAEv2"; ROOT="$base/users/hongyangd"; break; fi
done
: "${REPO:?could not find RAEv3/RAEv2 on the sensei mount}"
cd "$REPO"

mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/${JOB_NAME:-gfid}-node${RANK:-0}.log") 2>&1
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

NGPU="$(${PY} -c 'import torch;print(torch.cuda.device_count())')"
CONDS=("$@"); [ "${1:-}" = "all" ] && CONDS=(h1plain-noguid h1plain-ig178 encoder-noguid encoder-ig178)
[ ${#CONDS[@]} -eq 0 ] && { echo "usage: bash run_pluto_gfid.sh <h1plain-noguid|h1plain-ig178|encoder-noguid|encoder-ig178|all>"; exit 1; }

for cond in "${CONDS[@]}"; do
  CFG="configs/stage2/sampling/omnirae-eval-${cond}.yaml"
  [ -f "$CFG" ] || { echo "FATAL: no config $CFG"; exit 1; }
  export EXPERIMENT_NAME="omnirae-gfid-${cond}"
  echo "### $(date '+%F %T')  ${EXPERIMENT_NAME}  ngpu=${NGPU}  cfg=${CFG}"
  "$TR" --standalone --nproc_per_node="${NGPU:-8}" src/offline_eval.py --config "$CFG"
  CSV="results/stage2/eval/${EXPERIMENT_NAME}_ema.csv"
  echo "### $(date '+%F %T')  done ${cond}"
  if [ -f "$CSV" ]; then echo "### RESULT ($CSV):"; cat "$CSV"; echo; else echo "### WARN: no CSV at $CSV"; fi
done
echo "Done."
