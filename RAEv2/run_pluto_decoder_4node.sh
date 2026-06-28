#!/usr/bin/env bash
# MULTI-NODE entry for fine-tuning the stage-1 DECODER (train_decoder.py) on a Pluto
# job with N replicas x 8 GPUs (e.g. 4 nodes = 32 GPUs).
#
# Unlike the DiT scripts, train_decoder.py is fully config-driven (only takes --config);
# batch_size / lr / epochs / out_dir / init_from / wandb all live in the YAML.
# It auto-resumes from <out_dir>/ckpt_latest.pt; on the FIRST launch (no ckpt yet) it
# warm-starts weights from training.init_from and trains a fresh schedule from epoch 0.
#
# Pluto job: set replicas = NUM_NODES (e.g. 4), GPUs/replica = 8.
# Scripts field:
#   export WANDB_KEY=...            # optional (vault secret)
#   export NUM_NODES=4              # MUST match the replica count
#   bash /sensei-fs-3/users/hongyangd/RAEv3/RAEv2/run_pluto_decoder_4node.sh ft-plain
set -uo pipefail

for base in /sensei-fs-3 /mnt/remotes/sensei-fs-3; do
  if [ -d "$base/users/hongyangd/RAEv3/RAEv2" ]; then REPO="$base/users/hongyangd/RAEv3/RAEv2"; ROOT="$base/users/hongyangd"; break; fi
done
: "${REPO:?could not find RAEv3/RAEv2 on the sensei mount}"
cd "$REPO"

mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/${JOB_NAME:-decoder-ft}-node${RANK:-0}.log") 2>&1
echo "================ $(date '+%F %T')  host=$(hostname)  rank=${RANK:-0}  ================"

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
[ -x "$PY" ] || { echo "FATAL: portable env not found at $ROOT/rae_env"; exit 1; }

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
# HF streaming (WebDataset tar shards from the Hub, cached to localssd) for the 4-source mix.
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /mnt/localssd/raev2-wds-cache 2>/dev/null || true
export CKPT_KEEP_RECENT="${CKPT_KEEP_RECENT:-4}"   # keep recent N ckpt_ep*.pt (virtualized epochs -> many)
# wandb: train_decoder.py reads cfg.wandb.enabled; it needs an API key in the env.
[ -n "${WANDB_KEY:-}" ] && export WANDB_API_KEY="${WANDB_KEY}"
export WANDB_ENTITY="${WANDB_ENTITY:-uscgvl}"
export WANDB_PROJECT="${WANDB_PROJECT:-omnirae}"

NUM_NODES="${NUM_NODES:?set NUM_NODES to the job replica count (e.g. 4)}"
NPROC="${NUM_OF_GPUS:-8}"                       # GPUs per node (Pluto sets NUM_OF_GPUS)
NODE_RANK="${RANK:-0}"                          # Pluto sets RANK = replica/node index
MASTER="${MASTER_ADDR:-${JOB_NAME}-0}"          # rank-0 pod; fallback to <job>-0 hostname
MPORT="${MASTER_PORT:-29500}"

case "${1:-}" in
  ft-plain)      CFG=configs/stage1/decoder/ft-xcong-plain-k23-nodrop-4node.yaml ;;
  drop0-scratch) CFG=configs/stage1/decoder/ourpipe-drop0-k23-16ep-4node.yaml ;;
  general-4src)
    CFG=configs/stage1/decoder/omnirae-randomdrop-k23-general-4src.yaml
    # Node-count-agnostic: fix per-GPU batch=32 (OOM-safe with GAN, proven by drop0);
    # global batch = 32 x total_gpus scales with NUM_NODES. Keep 16 REAL epochs
    # (epochs/warmup/disc_start from the config, epoch-based); only sqrt-scale lr to the
    # actual global batch. steps/epoch = 69.7M / global_batch (varies with node count).
    eval "$("$PY" - "$NUM_NODES" "$NPROC" <<'PYEOF'
import sys, math
nodes, nproc = int(sys.argv[1]), int(sys.argv[2])
g = 32 * nodes * nproc                 # global batch (per-GPU 32)
lr = 2e-4 * math.sqrt(g / 512)
print(f"export BATCH_SIZE_OVERRIDE=32")
print(f"export LR_OVERRIDE={lr:.6e}")
print(f"# global_batch={g} lr={lr:.2e}  (16 real epochs; steps/epoch=69.7M/{g}={69_700_000//g})")
PYEOF
)"
    echo "### 4src auto: global=$((32*NUM_NODES*NPROC)) BATCH/GPU=32 LR=$LR_OVERRIDE epochs=16(real)"
    ;;
  *) echo "usage: NUM_NODES=4 bash run_pluto_decoder_4node.sh <ft-plain|drop0-scratch|general-4src>"; exit 1 ;;
esac

echo "### $(date '+%F %T')  decoder-ft  nodes=${NUM_NODES} node_rank=${NODE_RANK} nproc=${NPROC} master=${MASTER}:${MPORT} cfg=${CFG} wandb=$([ -n "${WANDB_API_KEY:-}" ] && echo on || echo off)"

exec "$TR" \
  --nnodes="${NUM_NODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC}" \
  --rdzv_backend=c10d \
  --rdzv_id="${JOB_NAME:-decoder-ft}-ftplain" \
  --rdzv_endpoint="${MASTER}:${MPORT}" \
  src/train_decoder.py \
  --config "$CFG"
