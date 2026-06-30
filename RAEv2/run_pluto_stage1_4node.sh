#!/usr/bin/env bash
# MULTI-NODE entry for reproducing the OFFICIAL RAEv2 stage-1 DECODER from scratch
# (train_stage1.py + RAE) on a Pluto job with N replicas x 8 GPUs (e.g. 4 nodes = 32 GPUs).
#
# train_stage1.py is the official model path (rae.py RAE: encoder forward_features adds the
# L23 token-mean surrogate + latent normalization). Same CLI as the DiT trainer
# (--config/--results-dir/--precision/--wandb/--epochs), torchrun + EXPERIMENT_NAME, and it
# auto-resumes from <results-dir>/<EXPERIMENT_NAME>. global_batch 512 / (NUM_NODES*8) per GPU;
# no grad-accum override (train_stage1 splits global_batch by world size directly).
#
# Pluto job: set replicas = NUM_NODES (e.g. 4), GPUs/replica = 8.
# Scripts field:
#   export WANDB_KEY=...            # optional (vault secret)
#   export NUM_NODES=4              # MUST match the replica count
#   bash /sensei-fs-3/users/hongyangd/RAEv3/RAEv2/run_pluto_stage1_4node.sh repro-k23
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
export CKPT_KEEP_RECENT="${CKPT_KEEP_RECENT:-4}"   # keep only 4 most recent ep-*.pt (~36GB);
                                                   # multi-node 16ep run = 125GB+ otherwise ->
                                                   # fills quota -> torch.save truncates -> crash-loop
export WANDB_ENTITY="${WANDB_ENTITY:-uscgvl}"
export WANDB_PROJECT="${WANDB_PROJECT:-omnirae}"
export WANDB_FRESH_RUN=1   # fresh wandb run each launch (avoids resume step-collision)

NUM_NODES="${NUM_NODES:?set NUM_NODES to the job replica count (e.g. 4)}"
NPROC="${NUM_OF_GPUS:-8}"                       # GPUs per node (Pluto sets NUM_OF_GPUS)
NODE_RANK="${RANK:-0}"                          # Pluto sets RANK = replica/node index
MASTER="${MASTER_ADDR:-${JOB_NAME}-0}"          # rank-0 pod; fallback to <job>-0 hostname
MPORT="${MASTER_PORT:-29500}"

case "${1:-}" in
  repro-k23) CFG=configs/stage1/training/repro-official-k23-16ep.yaml; BASE=repro-official-k23-16ep ;;
  nano-k23)  CFG=configs/stage1/training/repro-nano-k23-16ep.yaml;     BASE=repro-nano-k23-16ep ;;
  *) echo "usage: NUM_NODES=2 bash run_pluto_stage1_4node.sh <repro-k23|nano-k23>"; exit 1 ;;
esac
export EXPERIMENT_NAME="${BASE}-${NUM_NODES}node"

WANDB_FLAG=""; [ -n "${WANDB_KEY:-}" ] && WANDB_FLAG="--wandb"
echo "### $(date '+%F %T')  ${EXPERIMENT_NAME}  nodes=${NUM_NODES} node_rank=${NODE_RANK} nproc=${NPROC} master=${MASTER}:${MPORT} cfg=${CFG} wandb=${WANDB_FLAG:-off}"

exec "$TR" \
  --nnodes="${NUM_NODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC}" \
  --rdzv_backend=c10d \
  --rdzv_id="${JOB_NAME:-stage1-repro}-${BASE}" \
  --rdzv_endpoint="${MASTER}:${MPORT}" \
  --rdzv_conf=timeout=300 \
  src/train_stage1.py \
  --config "$CFG" \
  --results-dir "$ROOT/ckpt" \
  --precision bf16 ${WANDB_FLAG}
