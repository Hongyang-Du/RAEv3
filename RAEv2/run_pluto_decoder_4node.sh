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

# REPO = the repo THIS script lives in, so the old-code worktree (RAEv3_oldnorm) runs ITS
# OWN code + configs instead of hardcoding the main RAEv3/RAEv2. ROOT stays the shared
# sensei user dir (rae_env, dinov3, caches, out_dirs).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for base in /sensei-fs-3 /mnt/remotes/sensei-fs-3; do
  if [ -d "$base/users/hongyangd" ]; then ROOT="$base/users/hongyangd"; break; fi
done
: "${REPO:?could not resolve script dir}"; : "${ROOT:?could not find sensei user dir}"
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
export CKPT_KEEP_RECENT="${CKPT_KEEP_RECENT:-2}"   # keep recent N ckpt_ep*.pt (virtualized epochs -> many)
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-500}" # also save ckpt_latest every N steps -> survives Porter preemption (resume, not restart-from-zero)
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
  nano-drop)     CFG=configs/stage1/decoder/random-drop-layer-mls-plain-k23-nano.yaml ;;
  nano-drop-p005) CFG=configs/stage1/decoder/randomdrop-plain-k23-nano-p005-oldnorm.yaml ;; # old-regime retrain p_drop=0.05
  nano-drop-p01) CFG=configs/stage1/decoder/randomdrop-plain-k23-nano-p01-oldnorm.yaml ;;  # old-regime retrain p_drop=0.1
  nano-drop-p05) CFG=configs/stage1/decoder/randomdrop-plain-k23-nano-p05-oldnorm.yaml ;;  # old-regime retrain p_drop=0.5
  nano-drop-p07) CFG=configs/stage1/decoder/randomdrop-plain-k23-nano-p07-oldnorm.yaml ;;  # old-regime retrain p_drop=0.7
  nano-drop-p03) CFG=configs/stage1/decoder/randomdrop-plain-k23-nano-p03-oldnorm.yaml ;;  # old-regime p_drop=0.3 REPRO of the 0.294 anchor
  nano-drop-p002) CFG=configs/stage1/decoder/randomdrop-plain-k23-nano-p002-oldnorm.yaml ;;  # old-regime p_drop=0.02
  nano-drop-p09) CFG=configs/stage1/decoder/randomdrop-plain-k23-nano-p09-oldnorm.yaml ;;  # old-regime p_drop=0.9
  nano-drop-sched-0.9to0.1) CFG=configs/stage1/decoder/randomdrop-plain-k23-nano-sched-0.9to0.1-oldnorm.yaml ;;  # old-regime p_drop schedule 0.9->0.1
  nano-ft-k23) CFG=configs/stage1/decoder/ft-k23-nodrop-oldnorm.yaml ;;  # finetune omni->k23 (freeze/LoRA via env FREEZE_PREFIXES / LORA_R)
  nano-drop-k7-p03) CFG=configs/stage1/decoder/randomdrop-plain-k7-nano-p03-oldnorm.yaml ;;  # native k7 (7-layer) random-drop p_drop=0.3
  nano-drop-k7-p01) CFG=configs/stage1/decoder/randomdrop-plain-k7-nano-p01-oldnorm.yaml ;;  # native k7 (7-layer) random-drop p_drop=0.1
  nano-drop-p03-ganheavy) CFG=configs/stage1/decoder/randomdrop-plain-k23-nano-p03-ganheavy-oldnorm.yaml ;;  # k23 p_drop=0.3 w/ disc_weight 4.0 + disc_start=0
  nano-drop-p03-maskcond) CFG=configs/stage1/decoder/randomdrop-plain-k23-nano-p03-maskcond-oldnorm.yaml ;;  # Variant A mask conditioning (AdaLN), from scratch, anchor 16ep recipe
  nano-drop-p03-depthattn) CFG=configs/stage1/decoder/randomdrop-plain-k23-nano-p03-depthattn-oldnorm.yaml ;;  # Variant B depth-attn fusion, from scratch, anchor 16ep recipe
  nano-drop-k7-p03-depthattn) CFG=configs/stage1/decoder/randomdrop-plain-k7-nano-p03-depthattn-oldnorm.yaml ;;  # k7 (deep-only) depth-attn fusion, from scratch, anchor 16ep recipe
  jepa-k7-decoder) CFG=configs/stage1/decoder/jepa-k7-stage1-decoder-oldnorm.yaml ;;  # Stage-1: decoder on FROZEN JEPA k7 fusion (needs STAGE0_COMBINE env)
  jepa-k23-decoder) CFG=configs/stage1/decoder/jepa-k23-stage1-decoder-oldnorm.yaml ;;  # Stage-1: decoder on FROZEN JEPA k23 fusion (needs STAGE0_COMBINE env)
  rent-gandetach) CFG=configs/stage1/decoder/rent-k23-depthattn-softmax-gandetach-2node.yaml ;;  # JOINT fusion+decoder + semantic rent; GAN DETACHED from fusion. TRAINABLE fusion -> do NOT set STAGE0_COMBINE.
  rent-ganfusion) CFG=configs/stage1/decoder/rent-k23-depthattn-softmax-ganfusion-2node.yaml ;;  # same, but Phase-B GAN reaches fusion (ablation arm). do NOT set STAGE0_COMBINE.
  nano-drop-k7-p05-depthattn) CFG=configs/stage1/decoder/randomdrop-plain-k7-nano-p05-depthattn-oldnorm.yaml ;;  # k7 (deep-only) depth-attn fusion, p_drop=0.5, from scratch, anchor 16ep recipe
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
lr = 8e-4 * math.sqrt(g / 256)        # base: drop0/sigreg reference (8e-4 @ gb=256)
print(f"export BATCH_SIZE_OVERRIDE=32")
print(f"export LR_OVERRIDE={lr:.6e}")
print(f"# global_batch={g} lr={lr:.2e}  (16 real epochs; steps/epoch=69.7M/{g}={69_700_000//g})")
PYEOF
)"
    echo "### 4src auto: global=$((32*NUM_NODES*NPROC)) BATCH/GPU=32 LR=$LR_OVERRIDE epochs=16(real)"
    ;;
  general-4src-local)
    # Pre-download the 3 streaming sources to THIS node's localssd, then train from local.
    # Steady fast local reads -> no Pluto-autorecovery "hang" restarts (the streaming
    # variant stalls on cold HF reads and gets killed). imagenet stays on the shared FS.
    CFG=configs/stage1/decoder/omnirae-randomdrop-k23-general-4src-local.yaml
    DEST=/mnt/localssd/raev2-data
    mkdir -p "$DEST"
    echo "### $(date '+%F %T') downloading RAEv2 4src to $DEST (per-node, ~1.9TB, hf_transfer)..."
    HF_HUB_ENABLE_HF_TRANSFER=1 "$ROOT/rae_env/bin/hf" download nanovisionx/RAEv2-data \
      --repo-type dataset --local-dir "$DEST" \
      --include "rendertext-256/**" "scale-rae-256/**" \
                "blip3o-256/short-caption/**" "blip3o-256/long-caption/**"
      dl_rc=$?
    echo "### $(date '+%F %T') download exit=$dl_rc ; tars: rt=$(ls "$DEST"/rendertext-256/*.tar 2>/dev/null|wc -l) flux=$(ls "$DEST"/scale-rae-256/*.tar 2>/dev/null|wc -l) blip=$(ls "$DEST"/blip3o-256/*/*.tar 2>/dev/null|wc -l)"
    [ "$dl_rc" -ne 0 ] && { echo "FATAL: download failed"; exit 1; }
    eval "$("$PY" - "$NUM_NODES" "$NPROC" <<'PYEOF'
import sys, math
nodes, nproc = int(sys.argv[1]), int(sys.argv[2])
g = 32 * nodes * nproc
print(f"export BATCH_SIZE_OVERRIDE=32")
print(f"export LR_OVERRIDE={8e-4*math.sqrt(g/256):.6e}")
PYEOF
)"
    echo "### 4src-local: global=$((32*NUM_NODES*NPROC)) BATCH/GPU=32 LR=$LR_OVERRIDE epochs=16(real)"
    ;;
  general-4src-s3)
    # Same as general-4src-local but pulls the data from OUR S3 bucket (same AWS region as
    # Pluto -> fast, no HF rate-limits) instead of the HF Hub. Needs AWS creds in the job
    # env (or an instance role with s3:GetObject on the bucket).
    CFG=configs/stage1/decoder/omnirae-randomdrop-k23-general-4src-local.yaml
    DEST=/mnt/localssd/raev2-data
    S3=s3://hongyangd-raev2-backup/raev2-data
    mkdir -p "$DEST"
    echo "### $(date '+%F %T') syncing $S3 -> $DEST (per-node, ~1.9TB, AWS-internal)..."
    aws s3 sync "$S3" "$DEST" --only-show-errors
      dl_rc=$?
    echo "### $(date '+%F %T') s3 sync exit=$dl_rc ; tars: rt=$(ls "$DEST"/rendertext-256/*.tar 2>/dev/null|wc -l) flux=$(ls "$DEST"/scale-rae-256/*.tar 2>/dev/null|wc -l) blip=$(ls "$DEST"/blip3o-256/*/*.tar 2>/dev/null|wc -l)"
    [ "$dl_rc" -ne 0 ] && { echo "FATAL: s3 sync failed (creds? instance role?)"; exit 1; }
    eval "$("$PY" - "$NUM_NODES" "$NPROC" <<'PYEOF'
import sys, math
nodes, nproc = int(sys.argv[1]), int(sys.argv[2])
g = 32 * nodes * nproc
print(f"export BATCH_SIZE_OVERRIDE=32")
print(f"export LR_OVERRIDE={8e-4*math.sqrt(g/256):.6e}")
PYEOF
)"
    echo "### 4src-s3: global=$((32*NUM_NODES*NPROC)) BATCH/GPU=32 LR=$LR_OVERRIDE epochs=16(real)"
    ;;
  *) echo "usage: NUM_NODES=4 bash run_pluto_decoder_4node.sh <ft-plain|drop0-scratch|nano-drop|nano-drop-p005|nano-drop-p01|nano-drop-p05|nano-drop-p07|nano-drop-k7-p03|nano-drop-p03-ganheavy|nano-drop-p03-depthattn|nano-drop-k7-p03-depthattn|nano-drop-k7-p05-depthattn|nano-ft-k23|jepa-k7-decoder|jepa-k23-decoder|general-4src|general-4src-local|general-4src-s3>"; exit 1 ;;
esac

# Stage nanovisionx imagenet-256 from S3 -> node-local SSD (the /sensei-fs shared copy was
# deleted; the nano configs' data_dir points at $LSSD). Runs on EVERY node; skip if already
# staged. Needs AWS creds / instance-role with s3:GetObject on the bucket.
if [[ "$CFG" == *nano* || "$CFG" == *oldnorm* || "$CFG" == *rent* ]]; then
  LSSD=/mnt/localssd/imagenet-256
  if [ ! -f "$LSSD/imagenet-latents-images/dataset_info.json" ]; then
    echo "### $(date '+%F %T') staging nano imagenet -> $LSSD ..."
    mkdir -p "$LSSD"
    aws s3 sync s3://hongyangd-raev2-backup/raev2-data/imagenet-256/ "$LSSD/" \
      || { echo "### FATAL: S3 sync failed (need AWS creds/role on node)"; exit 1; }
    echo "### $(date '+%F %T') staged: $(du -sh "$LSSD" 2>/dev/null | cut -f1)"
  else
    echo "### nano imagenet already on local SSD ($LSSD)"
  fi
fi

# ROBUSTNESS: worker nodes wait for the master's rendezvous port to be OPEN before starting
# torchrun. Because each fresh pod re-stages ~236G to (ephemeral) local SSD, staging finishes
# at slightly different times per node; without this, a worker that finishes first starts
# torchrun, fails to reach the not-yet-up master store, its POD exits, and Pluto respawns a
# fresh pod that must re-stage 236G again -> the two nodes never rendezvous ("keeps reconnecting").
# Waiting here keeps the worker pod alive until the master is reachable. (bash /dev/tcp; no nc dep.)
if [ "${NUM_NODES}" -gt 1 ] && [ "${NODE_RANK}" -ne 0 ]; then
  echo "### $(date '+%F %T') rank ${NODE_RANK}: waiting for master ${MASTER}:${MPORT} ..."
  for _i in $(seq 1 360); do   # up to ~60 min (covers cold 236G stage on master)
    if (exec 3<>"/dev/tcp/${MASTER}/${MPORT}") 2>/dev/null; then exec 3>&- 3<&-; echo "### master up"; break; fi
    sleep 10
  done
fi

echo "### $(date '+%F %T')  decoder-ft  nodes=${NUM_NODES} node_rank=${NODE_RANK} nproc=${NPROC} master=${MASTER}:${MPORT} cfg=${CFG} wandb=$([ -n "${WANDB_API_KEY:-}" ] && echo on || echo off)"

exec "$TR" \
  --nnodes="${NUM_NODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC}" \
  --rdzv_backend=c10d \
  --rdzv_id="${JOB_NAME:-decoder-ft}-ftplain" \
  --rdzv_endpoint="${MASTER}:${MPORT}" \
  --rdzv_conf=timeout=2400 \
  src/train_decoder.py \
  --config "$CFG"
