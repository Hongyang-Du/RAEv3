#!/usr/bin/env bash
# ============================================================
#  Stage-2 DiT (DDT + internal guidance, flow matching) on a local stage-1
#  Usage:  bash run_train_dit.sh {raev2mls|nogate|dropmean}
#
#  Pipeline per variant:
#    1. compute latent stats (Welford mean/var of rae.encode over ImageNet)
#       if <stage1 outdir>/latent_stats.pt is missing
#    2. torchrun src/train.py with the matching stage-2 config
#
#  Latent: [B,1024,16,16] (frozen stage-1), x-prediction flow matching,
#  logit-normal t + shift 8, DDT-XL (28x1440 + 2x2048 head), gmuon lr 2e-4,
#  global batch 1024, bf16. Auto-resumes from ckpts_full/stage2/<EXPERIMENT_NAME>.
# ============================================================
set -euo pipefail

VARIANT=${1:?usage: run_train_dit.sh {raev2mls|nogate|dropmean}}

CONDA_ENV=/opt/conda/envs/rae
TORCHRUN=${CONDA_ENV}/bin/torchrun
PYTHON=${CONDA_ENV}/bin/python
cd "$(dirname "$(realpath "$0")")"

case "${VARIANT}" in
  raev2mls)
    CONFIG=configs/stage2/training/imagenet-dinov3l-k7-raev2mls.yaml
    S1_DIR=output_full/train_decoder_mls_raev2
    ;;
  nogate)
    CONFIG=configs/stage2/training/imagenet-dinov3l-k7-nogate-sigreg.yaml
    S1_DIR=output_full/train_decoder_mls_nogate_sigreg
    ;;
  dropmean)
    CONFIG=configs/stage2/training/imagenet-dinov3l-k7-dropmean-sigreg.yaml
    S1_DIR=output_full/train_decoder_mls_dropmean_sigreg
    ;;
  *) echo "unknown variant: ${VARIANT}"; exit 1 ;;
esac

# ── config ───────────────────────────────────────────────────
NGPU=8
DATA=/datasets/imagenet-256-full
RESULTS_DIR=ckpts_full/stage2
export EXPERIMENT_NAME=dit-${VARIANT}-k7
STATS=${S1_DIR}/latent_stats.pt
S1_CKPT=${S1_DIR}/ckpt_latest.pt
NUM_STATS_SAMPLES=250000
PRECISION=bf16

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=$(grep -A2 'api.wandb.ai' ~/.netrc | grep password | awk '{print $2}')
export WANDB_PROJECT=raev3-full
export WANDB_ENTITY=uscgvl
# ─────────────────────────────────────────────────────────────

[[ -f "${S1_CKPT}" ]] || { echo "stage-1 ckpt missing: ${S1_CKPT}"; exit 1; }
if pgrep -f "[s]rc/train_decoder_mls" > /dev/null; then
    echo "WARNING: a stage-1 training is still running — ${S1_CKPT} may not be final."
fi

echo "========================================================"
echo "  Stage-2 DiT  variant=${VARIANT}"
echo "  Config:   ${CONFIG}"
echo "  Stage-1:  ${S1_CKPT}"
echo "  Stats:    ${STATS} $( [[ -f ${STATS} ]] && echo '(exists)' || echo "(will compute, ${NUM_STATS_SAMPLES} samples)" )"
echo "  Output:   ${RESULTS_DIR}/${EXPERIMENT_NAME}"
echo "  wandb:    ${WANDB_ENTITY}/${WANDB_PROJECT}  run=${EXPERIMENT_NAME}"
echo "========================================================"

MASTER_PORT=$(${PYTHON} -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "Rendezvous port: ${MASTER_PORT}"

# ── 1. latent stats (skip if present) ────────────────────────
if [[ ! -f "${STATS}" ]]; then
    PYTORCH_ALLOC_CONF=expandable_segments:True ${TORCHRUN} --nproc_per_node=${NGPU} \
        --master-port="${MASTER_PORT}" \
        scripts/stage1/compute_latent_stats.py \
        --config       "${CONFIG}" \
        --data-dir     "${DATA}" \
        --output-path  "${STATS}" \
        --num-samples  "${NUM_STATS_SAMPLES}"
fi

# ── 2. DiT training ──────────────────────────────────────────
PYTORCH_ALLOC_CONF=expandable_segments:True ${TORCHRUN} --nproc_per_node=${NGPU} \
    --master-port="${MASTER_PORT}" \
    src/train.py \
    --config      "${CONFIG}" \
    --results-dir "${RESULTS_DIR}" \
    --precision   "${PRECISION}" \
    --wandb

echo ""
echo "Done. Checkpoints in ${RESULTS_DIR}/${EXPERIMENT_NAME}/checkpoints/"
