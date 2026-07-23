#!/usr/bin/env bash
# Single-node (8xH100) Pluto job: compute latent mean/var for the depthattn
# nano-p0.3 k23 stage-1 ckpt, so imagenet-dinov3l-depthattn-nano-p03-cls-k23.yaml's
# normalization_stat_path exists before the exp4 4-node DiT job (run_pluto_job_4node.sh)
# is launched -- that job loads this file on all 32 GPUs at startup and will
# crash-loop immediately if it's missing.
#
# Paste into the job's "Scripts" field (Pluto p5, 1 replica x 8 GPU):
#   bash /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2/run_pluto_stats_depthattn.sh
set -uo pipefail

for base in /sensei-fs-3 /mnt/remotes/sensei-fs-3; do
  if [ -d "$base/users/hongyangd/RAEv3_oldnorm/RAEv2" ]; then REPO="$base/users/hongyangd/RAEv3_oldnorm/RAEv2"; ROOT="$base/users/hongyangd"; break; fi
done
: "${REPO:?could not find RAEv3_oldnorm/RAEv2 on the sensei mount}"
cd "$REPO"

mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/${JOB_NAME:-depthattn-stats}-node${RANK:-0}.log") 2>&1
echo "================ $(date '+%F %T')  host=$(hostname)  rank=${RANK:-0}  ================"

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
[ -x "$PY" ] || { echo "FATAL: portable env not found at $ROOT/rae_env (run build_portable_env.sh first)"; exit 1; }

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

CFG=configs/stage2/training/imagenet-dinov3l-depthattn-nano-p03-cls-k23.yaml
DATA=/datasets/imagenet-256-full
OUT=/sensei-fs-3/users/hongyangd/ckpt/omni-randomdrop-plain-k23-nano-p0.3-depthattn/latent_stats.pt
NUM_SAMPLES=250000

if [ -f "$OUT" ]; then
  echo "### stats already exist at $OUT -- skipping (delete it first to recompute)"
  exit 0
fi

NGPU="$(${PY} -c 'import torch;print(torch.cuda.device_count())')"
echo "### $(date '+%F %T')  computing latent stats  cfg=$CFG  data=$DATA  out=$OUT  n=$NUM_SAMPLES  ngpu=$NGPU"

exec "$TR" --standalone --nproc_per_node="${NGPU:-8}" \
  scripts/stage1/compute_latent_stats.py \
  --config      "$CFG" \
  --data-dir    "$DATA" \
  --output-path "$OUT" \
  --num-samples "$NUM_SAMPLES"
