#!/usr/bin/env bash
# Entry script for a Pluto p5 (8xH100, single node) training job.
# Paste into the job's "Scripts" field:   bash /sensei-fs-3/users/hongyangd/RAEv3/RAEv2/run_pluto_job.sh exp2
# Uses the self-contained env on shared FS (no in-container pip/ssh needed).
set -uo pipefail

# --- locate repo on the sensei mount (path may be /sensei-fs-3 or /mnt/remotes/sensei-fs-3) ---
for base in /sensei-fs-3 /mnt/remotes/sensei-fs-3; do
  if [ -d "$base/users/hongyangd/RAEv3/RAEv2" ]; then REPO="$base/users/hongyangd/RAEv3/RAEv2"; ROOT="$base/users/hongyangd"; break; fi
done
: "${REPO:?could not find RAEv3/RAEv2 on the sensei mount}"
cd "$REPO"

# tee ALL output of this pod (incl. tracebacks) to the shared FS for easy debugging.
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/${JOB_NAME:-omnirae}-node${RANK:-0}.log") 2>&1
echo "================ $(date '+%F %T')  host=$(hostname)  rank=${RANK:-0}  ================"

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
[ -x "$PY" ] || { echo "FATAL: portable env not found at $ROOT/rae_env (run build_portable_env.sh first)"; exit 1; }

# --- offline encoder: local dinov3 code repo + local weights (no github needed) ---
export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export STAGE2_NO_EMA_CKPT=1   # don't store EMA in checkpoints (~1/3 smaller) — EMA unused downstream
export CKPT_KEEP_RECENT=6     # keep the 6 most recent checkpoints (rolling resume/fallback)
export CKPT_KEEP_EVERY=10     # plus keep every-10-epoch milestones (for gFID-vs-epoch eval)

# --- wandb (set WANDB_KEY as a job Vault Secret / env var) ---
export WANDB_ENTITY="${WANDB_ENTITY:-uscgvl}"
export WANDB_PROJECT="${WANDB_PROJECT:-omnirae}"
export WANDB_FRESH_RUN=1   # fresh wandb run each launch (avoids resume step-collision / crashed status)

case "${1:-}" in
  exp1) CFG=configs/stage2/training/imagenet-dinov3l-h1decoder-plain-cls-k23.yaml; export EXPERIMENT_NAME=omnirae-dit-h1-plain-cls-k23 ;;
  exp2) CFG=configs/stage2/training/imagenet-dinov3l-sigreg-cls-k23.yaml;          export EXPERIMENT_NAME=omnirae-dit-sigreg-cls-k23 ;;
  exp3) CFG=configs/stage2/training/imagenet-dinov3l-encoder-cls-k23.yaml;         export EXPERIMENT_NAME=omnirae-dit-encoder-cls-k23 ;;
  *) echo "usage: bash run_pluto_job.sh <exp1|exp2|exp3>"; exit 1 ;;
esac

WANDB_FLAG=""
[ -n "${WANDB_KEY:-}" ] && WANDB_FLAG="--wandb"   # only log to wandb if a key is provided

echo "### $(date '+%F %T')  ${EXPERIMENT_NAME}  py=$PY  cfg=$CFG  wandb=${WANDB_FLAG:-off}"
"$PY" -c "import torch;print('torch',torch.__version__,'cuda_avail',torch.cuda.is_available(),'n_gpu',torch.cuda.device_count())"

NGPU="$(${PY} -c 'import torch;print(torch.cuda.device_count())')"
exec "$TR" --standalone --nproc_per_node="${NGPU:-8}" src/train.py \
  --config "$CFG" \
  --results-dir "$ROOT/ckpt" \
  --precision bf16 ${WANDB_FLAG}
