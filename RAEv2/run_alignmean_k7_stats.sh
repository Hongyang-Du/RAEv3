#!/usr/bin/env bash
# Compute latent_stats.pt (Welford mean/var of the TRAINED alignmean-cls-k7-depthattn
# decoder's full-feed encode over ImageNet-256) for the Stage-2 DiT config
# imagenet-dinov3l-alignmean-cls-depthattn-k7.yaml. cls_surrogate:true + loss.align shift
# the latent off N(0,I), so the DiT needs this to standardize (normalization_stat_path in
# the config points here). k7 sibling of run_alignmean_k23_stats.sh.
#
# Single node, run ONCE before evaluating/continuing alignmean-k7-dit. Output:
#   /sensei-fs-3/users/hongyangd/ckpt/alignmean-cls-k7-depthattn/latent_stats.pt
#
#   NGPU=4 bash run_alignmean_k7_stats.sh
set -euo pipefail

for base in /sensei-fs-3 /mnt/remotes/sensei-fs-3; do
  if [ -d "$base/users/hongyangd/RAEv3_oldnorm/RAEv2" ]; then REPO="$base/users/hongyangd/RAEv3_oldnorm/RAEv2"; ROOT="$base/users/hongyangd"; break; fi
done
: "${REPO:?could not find RAEv3_oldnorm/RAEv2 on the sensei mount}"
cd "$REPO"

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
[ -x "$PY" ] || { echo "FATAL: portable env not found at $ROOT/rae_env"; exit 1; }

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

NGPU="${NGPU:-8}"
CONFIG=configs/stage2/training/imagenet-dinov3l-alignmean-cls-depthattn-k7.yaml
DATA="${DATA:-/mnt/localssd/imagenet-256}"
STATS="${STATS:-$ROOT/ckpt/alignmean-cls-k7-depthattn/latent_stats.pt}"
NUM_STATS_SAMPLES="${NUM_STATS_SAMPLES:-250000}"

[ -d "$DATA/imagenet-latents-images" ] || { echo "FATAL: imagenet-256 not staged at $DATA (expected $DATA/imagenet-latents-images)"; exit 1; }

if [[ -f "${STATS}" ]]; then
  echo "latent_stats already exists: ${STATS} (delete to recompute)"; exit 0
fi

MASTER_PORT=$(${PY} -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "========================================================"
echo "  alignmean-cls-k7-depthattn latent stats"
echo "  Config:  ${CONFIG}"
echo "  Data:    ${DATA}"
echo "  Samples: ${NUM_STATS_SAMPLES}   GPUs: ${NGPU}   port: ${MASTER_PORT}"
echo "  Output:  ${STATS}"
echo "========================================================"

${TR} --nproc_per_node=${NGPU} --master-port="${MASTER_PORT}" \
  scripts/stage1/compute_latent_stats.py \
  --config      "${CONFIG}" \
  --data-dir    "${DATA}" \
  --output-path "${STATS}" \
  --num-samples "${NUM_STATS_SAMPLES}"

echo "Done -> ${STATS}"
