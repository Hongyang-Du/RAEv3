#!/usr/bin/env bash
# ============================================================
#  SpatialAttnRes + SIGReg + GAN  Stage-1 training
#  Encoder:  DINOv3-L (frozen, layers 11,13,15,17,19,21,23)
#  Decoder:  ViT-XL (fine-tuned from pretrained)
#  Dataset:  partial ImageNet-256 (~93K images)
# ============================================================
set -euo pipefail

CONDA_ENV=/home/colligo/miniconda3/envs/rae
TORCHRUN=${CONDA_ENV}/bin/torchrun
PYTHON=${CONDA_ENV}/bin/python
SCRIPT=$(dirname "$(realpath "$0")")/src/train_attnres.py

cd "$(dirname "$(realpath "$0")")"

# ── config ───────────────────────────────────────────────────
NGPU=2
DATA=/home/colligo/data/imagenet-256/imagenet-256
OUT_DIR=output/train_attnres

EPOCHS=100
BATCH=64                 # per GPU; global = BATCH × NGPU
LR=2e-4
PRECISION=bf16           # bf16 saves ~50% memory, same accuracy on A100

SIGREG_W=0.1
LPIPS_W=1.0
DISC_WEIGHT=0.75
DISC_START=1             # epoch to start GAN (disc uses half-batch to save memory)

CKPT_EVERY=20            # save every 20 epochs, keep all
VAL_EVERY=500            # log val images every N steps
LOG_EVERY=50
VAL_IMAGE=assets/samples/sample_1.png  # fixed image to track reconstruction quality

WANDB=true
export WANDB_BASE_URL=https://adobesensei.wandb.io
export WANDB_API_KEY=$(grep -A2 'adobesensei.wandb.io' ~/.netrc | grep password | awk '{print $2}')
WANDB_PROJECT=raev3
WANDB_ENTITY=hongyangd
# ─────────────────────────────────────────────────────────────

echo "========================================================"
echo "  SpatialAttnRes Stage-1 Training"
echo "  GPUs:      ${NGPU}"
echo "  Batch:     ${BATCH}/GPU  (global: $((BATCH * NGPU)))"
echo "  Epochs:    ${EPOCHS}  |  Precision: ${PRECISION}"
echo "  Data:      ${DATA}"
echo "  Output:    ${OUT_DIR}"
echo "  wandb:     ${WANDB} (${WANDB_ENTITY}/${WANDB_PROJECT})"
echo "========================================================"
echo ""

WANDB_ARGS=""
if [[ "${WANDB}" == "true" ]]; then
    WANDB_ARGS="--wandb --wandb-project ${WANDB_PROJECT} --wandb-entity ${WANDB_ENTITY}"
fi

PYTORCH_ALLOC_CONF=expandable_segments:True ${TORCHRUN} --nproc_per_node=${NGPU} "${SCRIPT}" \
    --data        "${DATA}" \
    --out-dir     "${OUT_DIR}" \
    --epochs      "${EPOCHS}" \
    --batch-size  "${BATCH}" \
    --precision   "${PRECISION}" \
    --lr          "${LR}" \
    --sigreg-w    "${SIGREG_W}" \
    --lpips-w     "${LPIPS_W}" \
    --disc-weight "${DISC_WEIGHT}" \
    --disc-start  "${DISC_START}" \
    --ckpt-every  "${CKPT_EVERY}" \
    --val-every   "${VAL_EVERY}" \
    --log-every   "${LOG_EVERY}" \
    --val-image   "${VAL_IMAGE}" \
    ${WANDB_ARGS}

echo ""
echo "Done. Outputs in ${OUT_DIR}/"
