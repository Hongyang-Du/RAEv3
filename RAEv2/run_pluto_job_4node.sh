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

# tee ALL output of this pod (incl. tracebacks) to the shared FS so failures on any
# node are readable from anywhere. One file per pod: logs/<job>-node<RANK>.log
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/${JOB_NAME:-omnirae}-node${RANK:-0}.log") 2>&1
echo "================ $(date '+%F %T')  host=$(hostname)  rank=${RANK:-0}  ================"

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
[ -x "$PY" ] || { echo "FATAL: portable env not found at $ROOT/rae_env"; exit 1; }

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
# SAVE EMA in stage-2 checkpoints (EMA weights give ~0.2 lower gFID and are what the
# paper/official eval uses via model_utils' ema-preferred load). Adds ~3.5GB/ckpt.
# (was: export STAGE2_NO_EMA_CKPT=1  -> stripped EMA to save disk; disabled to match paper.)
export CKPT_KEEP_RECENT="${CKPT_KEEP_RECENT:-2}"   # keep most-recent N ep-*.pt
export CKPT_KEEP_EVERY="${CKPT_KEEP_EVERY:-10}"    # + every-K-epoch milestones
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-500}" # also overwrite ep-<epoch>.pt every N steps -> survive spare-capacity preemption (resume mid-epoch, not from step 0)
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
  exp1) CFG=configs/stage2/training/imagenet-dinov3l-h1decoder-plain-cls-k23.yaml; BASE=omnirae-dit-h1-plain-cls-k23-ema ;;
  exp2) CFG=configs/stage2/training/imagenet-dinov3l-sigreg-cls-k23.yaml;          BASE=omnirae-dit-sigreg-cls-k23-ema ;;
  exp3) CFG=configs/stage2/training/imagenet-dinov3l-encoder-cls-k23.yaml;         BASE=omnirae-dit-encoder-cls-k23-ema ;;
  exp4) CFG=configs/stage2/training/imagenet-dinov3l-h1decoder-plain-cls-k7.yaml;  BASE=omnirae-dit-h1-plain-cls-k7-ema ;;  # k7 h1, evanarlian data, k23 decoder
  exp5) CFG=configs/stage2/training/imagenet-dinov3l-h1decoder-raev2k23.yaml;      BASE=omnirae-dit-h1-raev2k23-ema ;;       # ABLATION: h1 on RAEv2 K=23 decoder (repro-nano-k23, cls off)
  exp6) CFG=configs/stage2/training/imagenet-dinov3l-encoder-cls-k7.yaml;          BASE=omnirae-dit-encoder-cls-k7-ema ;;    # k7 encoder counterpart of exp3 (7-layer subset of the k23 decoder)
  *) echo "usage: NUM_NODES=4 bash run_pluto_job_4node.sh <exp1|exp2|exp3|exp4|exp5|exp6>"; exit 1 ;;
esac
export EXPERIMENT_NAME="${BASE}-${NUM_NODES}node"   # SEPARATE folder from the single-node run

# Stage the run's imagenet-256 to node-local SSD for fast training reads (worth the
# per-launch copy vs reading off NFS every step). Prefer S3, fall back to the sensei-fs
# mount. Skips if already staged on this node.
#   exp1 (k23-h1), exp3 (encoder), exp4 (k7-h1) all train on nanovisionx now
#   (h1 DiTs sit on the 2e-4 random-drop nano decoder -> use its own data).
#   FB (sensei-fs) is gone after the local nano delete, so this relies on S3 on the node.
case "${1:-}" in
  exp1|exp3|exp4|exp5|exp6) LSSD=/mnt/localssd/imagenet-256; S3SRC=s3://hongyangd-raev2-backup/raev2-data/imagenet-256/; FB="$ROOT/data/imagenet-256" ;;
  *)              LSSD="" ;;
esac
if [ -n "$LSSD" ]; then
  if [ ! -f "$LSSD/imagenet-latents-images/dataset_info.json" ]; then
    echo "### $(date '+%F %T') staging $S3SRC -> $LSSD ..."
    mkdir -p "$LSSD"
    aws s3 sync "$S3SRC" "$LSSD/" \
      || rsync -a "$FB/" "$LSSD/"
    echo "### $(date '+%F %T') staged: $(du -sh "$LSSD" 2>/dev/null | cut -f1)"
  else
    echo "### imagenet already on local SSD ($LSSD)"
  fi
fi

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
