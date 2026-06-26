#!/usr/bin/env bash
# MULTI-NODE entry for a Pluto job with N replicas x 8 GPUs (e.g. 4 nodes = 32 GPUs).
# Same global batch 1024 (grad_accum forced to 1 -> 32/GPU at 32 GPUs); just ~Nx faster.
# Saves to a SEPARATE folder (EXPERIMENT_NAME suffixed with -<NUM_NODES>node) so it does
# NOT collide with any single-node run already in progress.
#
# Pluto job: set replicas = NUM_NODES (e.g. 4), GPUs/replica = 8.
# Scripts field:
#   export WANDB_KEY=...            # optional (vault secret)
#   export NUM_NODES=4              # MUST match the replica count
#   bash /sensei-fs-3/users/hongyangd/RAEv3/RAEv2/run_pluto_job_4node.sh exp3
set -uo pipefail

for base in /sensei-fs-3 /mnt/remotes/sensei-fs-3; do
  if [ -d "$base/users/hongyangd/RAEv3/RAEv2" ]; then REPO="$base/users/hongyangd/RAEv3/RAEv2"; ROOT="$base/users/hongyangd"; break; fi
done
: "${REPO:?could not find RAEv3/RAEv2 on the sensei mount}"
cd "$REPO"

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
[ -x "$PY" ] || { echo "FATAL: portable env not found at $ROOT/rae_env"; exit 1; }

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export STAGE2_NO_EMA_CKPT=1
export CKPT_KEEP_RECENT=6
export CKPT_KEEP_EVERY=10
export WANDB_ENTITY="${WANDB_ENTITY:-uscgvl}"
export WANDB_PROJECT="${WANDB_PROJECT:-omnirae}"
export WANDB_FRESH_RUN=1   # fresh wandb run each launch (avoids resume step-collision / crashed status)

NUM_NODES="${NUM_NODES:?set NUM_NODES to the job replica count (e.g. 4)}"
NPROC="${NUM_OF_GPUS:-8}"                       # GPUs per node (Pluto sets NUM_OF_GPUS)
# keep global batch 1024 with <=64 micro/GPU: accum = ceil(1024 / (total_gpus * 64))
TOTAL_GPUS=$(( NUM_NODES * NPROC ))
export GRAD_ACCUM_OVERRIDE=$(( (1024 + TOTAL_GPUS*64 - 1) / (TOTAL_GPUS*64) ))
[ "${GRAD_ACCUM_OVERRIDE}" -lt 1 ] && export GRAD_ACCUM_OVERRIDE=1
NODE_RANK="${RANK:-0}"                          # Pluto sets RANK = replica/node index
MASTER="${MASTER_ADDR:-${JOB_NAME}-0}"          # rank-0 pod; fallback to <job>-0 hostname
MPORT="${MASTER_PORT:-29500}"

case "${1:-}" in
  exp1) CFG=configs/stage2/training/imagenet-dinov3l-h1decoder-plain-cls-k23.yaml; BASE=omnirae-dit-h1-plain-cls-k23 ;;
  exp2) CFG=configs/stage2/training/imagenet-dinov3l-sigreg-cls-k23.yaml;          BASE=omnirae-dit-sigreg-cls-k23 ;;
  exp3) CFG=configs/stage2/training/imagenet-dinov3l-encoder-cls-k23.yaml;         BASE=omnirae-dit-encoder-cls-k23 ;;
  *) echo "usage: NUM_NODES=4 bash run_pluto_job_4node.sh <exp1|exp2|exp3>"; exit 1 ;;
esac
export EXPERIMENT_NAME="${BASE}-${NUM_NODES}node"   # SEPARATE folder from the single-node run

WANDB_FLAG=""; [ -n "${WANDB_KEY:-}" ] && WANDB_FLAG="--wandb"
echo "### $(date '+%F %T')  ${EXPERIMENT_NAME}  nodes=${NUM_NODES} node_rank=${NODE_RANK} nproc=${NPROC} master=${MASTER}:${MPORT} cfg=${CFG} wandb=${WANDB_FLAG:-off}"

exec "$TR" \
  --nnodes="${NUM_NODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC}" \
  --rdzv_backend=c10d \
  --rdzv_id="${JOB_NAME:-omnirae}-${BASE}" \
  --rdzv_endpoint="${MASTER}:${MPORT}" \
  src/train.py \
  --config "$CFG" \
  --results-dir "$ROOT/ckpt" \
  --precision bf16 ${WANDB_FLAG}
