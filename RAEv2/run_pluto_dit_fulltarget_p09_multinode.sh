#!/usr/bin/env bash
# ============================================================
#  MULTI-NODE entry for DiT-B DECOUPLED FULL TARGET, p_drop=0.9.
#  Per-node entry: EVERY Pluto replica runs this exact script; torchrun forms one
#  DDP group of NUM_NODES x NUM_OF_GPUS ranks via c10d rendezvous on the rank-0 pod.
#
#  Recipe is byte-identical to the single-node p0.9 run -- ONLY the node count changes:
#    * config global_batch_size=2048 is the invariant. train.py derives
#      micro = 2048/(world_size*grad_accum_steps), so global batch stays 2048 and the
#      per-GPU micro-batch just shrinks with more nodes (256/GPU@1node -> 64@4 -> 32@8).
#      => NO GRAD_ACCUM_OVERRIDE; leave grad_accum_steps=1 as in the YAML.
#      Constraint: 2048 % (NUM_NODES*NUM_OF_GPUS) == 0 (holds for 1,2,4,8,16 nodes @8gpu).
#    * decoder is STILL the p0.3 stage-1 decoder (config stage1_ckpt_path); only the
#      combine p_drop (0.9) + DiT out_dir differ. transport.decoupled_full_target=true.
#    * latent_stats.pt is already precomputed at the config's absolute
#      normalization_stat_path (.../p0.9/latent_stats.pt) -> no stats step here.
#
#  Writes to EXPERIMENT_NAME=dit-b-omni-randomdrop-fulltarget-plain-k7-nano-p0.9 (the dir
#  that already holds latent_stats.pt). Auto-resumes from its ckpt_latest.pt, so a
#  preemption just resumes. To keep node-count runs in a SEPARATE folder, export
#  EXPERIMENT_SUFFIX=-4node (appended to the name).
#
#  Pluto job: replicas = NUM_NODES (e.g. 4), GPUs/replica = 8. Scripts field:
#    export WANDB_KEY=...            # optional (vault secret); omit -> wandb off
#    export NUM_NODES=4             # MUST match the replica count
#    bash /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2/run_pluto_dit_fulltarget_p09_multinode.sh
# ============================================================
set -uo pipefail

for base in /sensei-fs-3 /mnt/remotes/sensei-fs-3; do
  if [ -d "$base/users/hongyangd/RAEv3_oldnorm/RAEv2" ]; then REPO="$base/users/hongyangd/RAEv3_oldnorm/RAEv2"; ROOT="$base/users/hongyangd"; break; fi
done
: "${REPO:?could not find RAEv3_oldnorm/RAEv2 on the sensei mount}"
cd "$REPO"

# tee ALL output of this pod (incl. tracebacks) to the shared FS so a failure on any node
# is readable from anywhere. One file per pod: logs/<job>-p09-node<RANK>.log
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/${JOB_NAME:-dit-fulltarget-p09}-node${RANK:-0}.log") 2>&1
echo "================ $(date '+%F %T')  host=$(hostname)  rank=${RANK:-0}  ================"

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
[ -x "$PY" ] || { echo "FATAL: portable env not found at $ROOT/rae_env"; exit 1; }

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
# one-time NCCL transport lines at init (IB/EFA vs TCP fallback); no per-step flooding.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}"
export CKPT_KEEP_RECENT="${CKPT_KEEP_RECENT:-2}"
export CKPT_KEEP_EVERY="${CKPT_KEEP_EVERY:-10}"
# roll ckpt_latest.pt every N optimizer steps so a preemption (jobs restart ~every 1.5h)
# loses at most N steps, not the whole epoch.
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-500}"
export WANDB_ENTITY="${WANDB_ENTITY:-uscgvl}"
export WANDB_PROJECT="${WANDB_PROJECT:-omnirae}"
export WANDB_FRESH_RUN=1   # fresh wandb run each launch (avoids resume step-collision)

NUM_NODES="${NUM_NODES:?set NUM_NODES to the job replica count (e.g. 4)}"
NPROC="${NUM_OF_GPUS:-8}"                       # GPUs per node (Pluto sets NUM_OF_GPUS)
NODE_RANK="${RANK:-0}"                          # Pluto sets RANK = replica/node index
MASTER="${MASTER_ADDR:-${JOB_NAME}-0}"          # rank-0 pod; fallback to <job>-0 hostname
MPORT="${MASTER_PORT:-29500}"
TOTAL_GPUS=$(( NUM_NODES * NPROC ))

CFG="configs/stage2/training/imagenet-dinov3l-omni-randomdrop-fulltarget-plain-k7-nano-p09-ditb.yaml"
NAME="dit-b-omni-randomdrop-fulltarget-plain-k7-nano-p0.9${EXPERIMENT_SUFFIX:-}"
export EXPERIMENT_NAME="$NAME"
DECODER="$ROOT/ckpt/omni-randomdrop-plain-k7-nano-p0.3/ckpt_latest.pt"
STATS="$ROOT/ckpt/dit-b-omni-randomdrop-fulltarget-plain-k7-nano-p0.9/latent_stats.pt"

# global_batch_size (2048) must be divisible by the world size, else train.py asserts.
if [ $(( 2048 % TOTAL_GPUS )) -ne 0 ]; then
  echo "FATAL: global_batch_size 2048 not divisible by total GPUs ${TOTAL_GPUS} (NUM_NODES=${NUM_NODES} x NPROC=${NPROC}). Use 1,2,4,8,16 nodes @8gpu."; exit 1
fi
[ -f "$DECODER" ] || { echo "FATAL: decoder ckpt missing: $DECODER (config's stage1_ckpt_path)"; exit 1; }
[ -f "$STATS" ]   || { echo "FATAL: latent_stats missing: $STATS (precompute once before multi-node launch)"; exit 1; }

# Stage the ~236GB imagenet-256 arrow set to node-local SSD (NOT shared across nodes).
# The p09 config's data_dir is the ABSOLUTE /mnt/localssd/imagenet-256, so no symlink is
# needed -- just make sure that path is populated on THIS node. Sentinel: imagenet-latents-images.
LSSD=/mnt/localssd/imagenet-256
S3=s3://hongyangd-raev2-backup/raev2-data/imagenet-256
if [ ! -d "$LSSD/imagenet-latents-images" ]; then
  # LOUD PREFLIGHT before the (silent) bulk sync so a creds/role or reachability problem
  # surfaces IMMEDIATELY with a clear, flushed message -- rather than a healthy-looking
  # "--only-show-errors" sync that prints nothing until it either finishes or the pod is
  # killed (which is why the shared-FS tee'd log looked empty after "staging..."). If this
  # allocation lacks the S3 instance-role, `aws sts`/`aws s3 ls` fail here, fast and clear.
  echo "### $(date '+%F %T') S3 preflight: whoami + bucket reachability ..."
  aws sts get-caller-identity 2>&1 | sed 's/^/###   sts: /' || true
  if ! timeout 60 aws s3 ls "$S3/imagenet-latents-images/" >/dev/null 2>/tmp/_s3err; then
    echo "### FATAL: cannot list $S3 -- this allocation likely lacks the S3 instance-role/creds."
    echo "###        aws error was:"; sed 's/^/###   /' /tmp/_s3err
    echo "###        Fix: submit under an S3-enabled Pluto project/allocation (the one the"
    echo "###        alignmean-k23 job used), or attach a role with s3:GetObject on the bucket."
    exit 1
  fi
  echo "### $(date '+%F %T') preflight OK; staging imagenet-256 -> $LSSD (~236GB, AWS-internal)..."
  mkdir -p "$LSSD"
  aws s3 sync "$S3" "$LSSD" --only-show-errors \
    || { echo "### FATAL: S3 sync failed mid-transfer (see aws errors above)"; exit 1; }
  echo "### $(date '+%F %T') staged: $(du -sh "$LSSD" 2>/dev/null | cut -f1)"
else
  echo "### imagenet-256 already on local SSD ($LSSD)"
fi

WANDB_FLAG=""; [ -n "${WANDB_KEY:-}" ] && WANDB_FLAG="--wandb"
echo "### $(date '+%F %T')  ${EXPERIMENT_NAME}  nodes=${NUM_NODES} node_rank=${NODE_RANK} nproc=${NPROC} total_gpus=${TOTAL_GPUS} master=${MASTER}:${MPORT} cfg=${CFG} wandb=${WANDB_FLAG:-off}"

# PREFLIGHT: non-rank-0 pods fail fast on an unreachable rendezvous master instead of
# hanging the full join_timeout (a replaced rank-0 pod changes its hostname).
MASTER_WAIT_SECS="${MASTER_WAIT_SECS:-300}"
if [ "${NODE_RANK}" != "0" ]; then
  echo "### $(date '+%F %T') waiting up to ${MASTER_WAIT_SECS}s for rendezvous master '${MASTER}' to resolve..."
  __deadline=$(( $(date +%s) + MASTER_WAIT_SECS ))
  until getent hosts "${MASTER}" >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "${__deadline}" ]; then
      echo "### FATAL: rendezvous master '${MASTER}' never resolved in ${MASTER_WAIT_SECS}s"
      echo "###        (MASTER_ADDR=${MASTER_ADDR:-<unset>}, JOB_NAME=${JOB_NAME:-<unset>}, RANK=${NODE_RANK})."
      echo "###        rank-0 was likely preempted/replaced. Enable 'Auto Requeue on Preemption'"
      echo "###        so the whole job requeues together, and/or use a quota-backed allocation."
      exit 1
    fi
    sleep 5
  done
  echo "### $(date '+%F %T') master '${MASTER}' resolved: $(getent hosts "${MASTER}" | awk '{print $1}' | tr '\n' ' ')"
fi

exec "$TR" \
  --nnodes="${NUM_NODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC}" \
  --max-restarts=3 \
  --rdzv_backend=c10d \
  --rdzv_id="${JOB_NAME:-dit-fulltarget-p09}-${NAME}" \
  --rdzv_endpoint="${MASTER}:${MPORT}" \
  --rdzv_conf=join_timeout=600 \
  src/train.py \
  --config "$CFG" \
  --results-dir "$ROOT/ckpt" \
  --precision bf16 ${WANDB_FLAG}
