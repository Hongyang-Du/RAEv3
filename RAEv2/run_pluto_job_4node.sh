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
#   bash /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2/run_pluto_job_4node.sh exp3
set -uo pipefail

for base in /sensei-fs-3 /mnt/remotes/sensei-fs-3; do
  if [ -d "$base/users/hongyangd/RAEv3_oldnorm/RAEv2" ]; then REPO="$base/users/hongyangd/RAEv3_oldnorm/RAEv2"; ROOT="$base/users/hongyangd"; break; fi
done
: "${REPO:?could not find RAEv3_oldnorm/RAEv2 on the sensei mount}"
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
# DIAGNOSTIC (read-only): print the NCCL transport it picks at init so we can tell
# whether inter-node collectives use IB/EFA RDMA or fall back to slow TCP sockets.
# SUBSYS=INIT,NET keeps it to one-time setup lines (no per-step flooding). Remove
# once the ~3.3s/step multi-node slowdown is understood.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}"
# DIAGNOSTIC: rank0 prints per-step breakdown (data-wait / fwd / bwd+all-reduce / opt)
# for gstep 5-35 (see src/stage2/engine.py), to locate the cached-latent slowdown.
# Remove once understood.
export STEP_TIMING="${STEP_TIMING:-1}"
export STAGE2_NO_EMA_CKPT=1
export CKPT_KEEP_RECENT=6
export CKPT_KEEP_EVERY=10
# roll a ckpt_latest.pt every N optimizer steps so a preemption (jobs here restart ~every
# 1.5h) loses at most N steps, not the whole epoch. 500 * ~3.3s/step ~= 28min << restart
# period. If the per-step slowdown gets fixed (~0.5s/step), raise this so the 3.5GB rank0
# write stays a small fraction of wall time.
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-500}"
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
  exp1) CFG=configs/stage2/training/imagenet-dinov3l-h1decoder-plain-cls-k23.yaml;    BASE=omnirae-dit-h1-plain-cls-k23 ;;
  exp2) CFG=configs/stage2/training/imagenet-dinov3l-sigreg-cls-k23.yaml;             BASE=omnirae-dit-sigreg-cls-k23 ;;
  exp3) CFG=configs/stage2/training/imagenet-dinov3l-encoder-cls-k23.yaml;            BASE=omnirae-dit-encoder-cls-k23 ;;
  exp4) CFG=configs/stage2/training/imagenet-dinov3l-depthattn-nano-p03-cls-k23.yaml;             BASE=omnirae-dit-depthattn-cls-k23 ;;  # DepthAttnCombine (Variant B), drop:false = full-k23 latent, on-the-fly encode
  exp5) CFG=configs/stage2/training/imagenet-dinov3l-depthattn-nano-p03-cls-k23-cachedlatent.yaml; BASE=omnirae-dit-depthattn-cls-k23-cachedlatent ;;  # same as exp4 but reads precomputed latents (scripts/stage1/precompute_latents.py) -- needs /mnt/localssd/latents-depthattn-k23-nano-p03 staged on every node first
  *) echo "usage: NUM_NODES=4 bash run_pluto_job_4node.sh <exp1|exp2|exp3|exp4|exp5>"; exit 1 ;;
esac
export EXPERIMENT_NAME="${BASE}-${NUM_NODES}node"   # SEPARATE folder from the single-node run

# cachedlatent configs read from local SSD (src/data/latent_cache_dataset.py partitions
# shards by DDP rank, which spans ALL nodes) -> every node needs its OWN full copy, local
# SSD is not shared across nodes. Runs on EVERY node (this script IS the per-node entry);
# skip if already staged. ~656GB (bf16, 1.28M x 1024x16x16 samples) -> check node disk
# budget before adding more variants here. Needs AWS creds/instance-role with s3:GetObject
# on the bucket (same assumption run_pluto_decoder_4node.sh's imagenet staging makes).
if [[ "$CFG" == *cachedlatent* ]]; then
  LSSD=/mnt/localssd/latents-depthattn-k23-nano-p03
  S3=s3://hongyangd-raev2-backup/raev2-data/latents-depthattn-k23-nano-p03
  if [ ! -f "$LSSD/train/manifest.json" ]; then
    echo "### $(date '+%F %T') staging depthattn latent cache -> $LSSD (~656GB, AWS-internal)..."
    mkdir -p "$LSSD"
    aws s3 sync "$S3" "$LSSD" --only-show-errors \
      || { echo "### FATAL: S3 sync failed (need AWS creds/role on node)"; exit 1; }
    echo "### $(date '+%F %T') staged: $(du -sh "$LSSD" 2>/dev/null | cut -f1)"
  else
    echo "### depthattn latent cache already on local SSD ($LSSD)"
  fi
else
  # non-cachedlatent configs (exp1-4) encode on-the-fly from the raw imagenet-256 arrow
  # set. Their stage-2 configs use a RELATIVE data_dir (./data/imagenet-256), so stage the
  # ~236GB arrow set from S3 to node-local SSD (not shared across nodes) and point
  # ./data/imagenet-256 at it via symlink. Skip if already staged. Same AWS creds/role
  # assumption as the cachedlatent branch. imagenet_hf_dataset.py reads the train split at
  # <data_dir>/imagenet-latents-images, so that subdir's presence is the "staged" sentinel.
  LSSD=/mnt/localssd/imagenet-256
  S3=s3://hongyangd-raev2-backup/raev2-data/imagenet-256
  if [ ! -d "$LSSD/imagenet-latents-images" ]; then
    echo "### $(date '+%F %T') staging imagenet-256 -> $LSSD (~236GB, AWS-internal)..."
    mkdir -p "$LSSD"
    aws s3 sync "$S3" "$LSSD" --only-show-errors \
      || { echo "### FATAL: S3 sync failed (need AWS creds/role on node)"; exit 1; }
    echo "### $(date '+%F %T') staged: $(du -sh "$LSSD" 2>/dev/null | cut -f1)"
  else
    echo "### imagenet-256 already on local SSD ($LSSD)"
  fi
  # config's relative ./data/imagenet-256 -> node-local staged copy (each node its own)
  mkdir -p "$REPO/data"
  ln -sfn "$LSSD" "$REPO/data/imagenet-256"
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
  --rdzv_conf=join_timeout=1800 \
  src/train.py \
  --config "$CFG" \
  --results-dir "$ROOT/ckpt" \
  --precision bf16 ${WANDB_FLAG}
