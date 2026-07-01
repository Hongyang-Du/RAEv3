#!/usr/bin/env bash
# MULTI-NODE end-to-end joint training (SIGReg projector + decoder + DiT) via
# train_e2e_sigreg_dit.py (config-driven, takes --config). DDP across nodes.
# Auto-resumes from <out-dir>/ckpt_latest.pt (out-dir is set in the YAML).
#
# Pluto job: replicas = NUM_NODES (e.g. 2), GPUs/replica = 8. Scripts field:
#   export WANDB_API_KEY=...          # the e2e configs set wandb: true
#   export NUM_NODES=2
#   bash /sensei-fs-3/users/hongyangd/RAEv3/RAEv2/run_pluto_e2e_4node.sh nano-dit-full-dec-drop
set -uo pipefail

for base in /sensei-fs-3 /mnt/remotes/sensei-fs-3; do
  if [ -d "$base/users/hongyangd/RAEv3/RAEv2" ]; then REPO="$base/users/hongyangd/RAEv3/RAEv2"; ROOT="$base/users/hongyangd"; break; fi
done
: "${REPO:?could not find RAEv3/RAEv2 on the sensei mount}"
cd "$REPO"

mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/${JOB_NAME:-e2e}-node${RANK:-0}.log") 2>&1
echo "================ $(date '+%F %T')  host=$(hostname)  rank=${RANK:-0}  ================"

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
[ -x "$PY" ] || { echo "FATAL: portable env not found at $ROOT/rae_env"; exit 1; }

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Preemption resilience on spare-capacity: save ckpt_latest every N steps so an
# autorecovery reclaim (often <1 epoch apart) resumes instead of restarting the epoch.
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-500}"
export WANDB_ENTITY="${WANDB_ENTITY:-uscgvl}"
export WANDB_PROJECT="${WANDB_PROJECT:-raev3-full}"
# train_e2e_sigreg_dit.py does wandb.init() (config wandb: true), which needs
# WANDB_API_KEY. Map the WANDB_KEY the Pluto Scripts export to it; if neither is
# set, run wandb offline so a missing key never crashes the job.
export WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_KEY:-}}"
[ -z "${WANDB_API_KEY}" ] && export WANDB_MODE=offline

NUM_NODES="${NUM_NODES:?set NUM_NODES to the job replica count (e.g. 2)}"
NPROC="${NUM_OF_GPUS:-8}"                       # GPUs per node (Pluto sets NUM_OF_GPUS)
NODE_RANK="${RANK:-0}"                          # Pluto sets RANK = replica/node index
MASTER="${MASTER_ADDR:-${JOB_NAME}-0}"          # rank-0 pod
MPORT="${MASTER_PORT:-29500}"

case "${1:-}" in
  nano-dit-full-dec-drop) CFG=configs/e2e/e2e-dit-full-dec-drop-nano.yaml; BASE=e2e-sigreg-dit-full-dec-drop-nano ;;
  *) echo "usage: NUM_NODES=2 bash run_pluto_e2e_4node.sh <nano-dit-full-dec-drop>"; exit 1 ;;
esac

# Stage nanovisionx imagenet-256 from S3 to node-local SSD (local /sensei-fs copy was
# deleted; the config reads /mnt/localssd/imagenet-256). Skips if already staged.
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

echo "### $(date '+%F %T')  ${BASE}  nodes=${NUM_NODES} node_rank=${NODE_RANK} nproc=${NPROC} master=${MASTER}:${MPORT} cfg=${CFG}"

exec "$TR" \
  --nnodes="${NUM_NODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC}" \
  --rdzv_backend=c10d \
  --rdzv_id="${JOB_NAME:-e2e}-${BASE}" \
  --rdzv_endpoint="${MASTER}:${MPORT}" \
  --rdzv_conf=timeout=300 \
  src/train_e2e_sigreg_dit.py --config "$CFG"
