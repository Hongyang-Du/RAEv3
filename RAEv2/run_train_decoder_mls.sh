#!/usr/bin/env bash
# ============================================================
#  raev2 MLS decoder training  (src/train_decoder_mls.py)
#  Encoder:  DINOv3-L (frozen, layers 11,13,15,17,19,21,23)
#  Combine:  raev2 dinov3mls multi-layer sum (frozen, no params)
#  Decoder:  ViT-XL (trained from scratch) -- the ONLY trainable module
#  Dataset:  partial ImageNet-256 (~93K images)
#
#  Identical recipe to run_train_attnres.sh, except the multi-layer
#  combination is raev2's fixed MLS instead of the learned SpatialAttnRes
#  (so there is no SIGReg and only the decoder trains).
# ============================================================
set -euo pipefail

CONDA_ENV=/opt/conda/envs/rae
TORCHRUN=${CONDA_ENV}/bin/torchrun
PYTHON=${CONDA_ENV}/bin/python
SCRIPT=$(dirname "$(realpath "$0")")/src/train_decoder_mls.py

cd "$(dirname "$(realpath "$0")")"

# -- config ---------------------------------------------------
NGPU=8
DATA=/datasets/imagenet-256
OUT_DIR=output/train_decoder_mls_raev2

EPOCHS=50
BATCH=32                 # per GPU; global = BATCH x NGPU
LR=8e-4
PRECISION=bf16           # bf16 saves ~50% memory, same accuracy on A100
LAYERS="11 13 15 17 19 21 23"   # DINOv3-L layers to combine (K7); e.g. "23" = last layer only

LPIPS_W=1.0
DISC_WEIGHT=0.75
DISC_START=1             # epoch to start GAN (disc uses half-batch to save memory)

CKPT_EVERY=10           # keep a permanent ckpt every N epochs (ckpt_latest always saved)
VAL_EVERY=500            # log val images every N steps
LOG_EVERY=50
VAL_IMAGE=assets/samples/sample_1.png  # fixed images (dir globbed) to track recon quality

WANDB=true
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=$(grep -A2 'api.wandb.ai' ~/.netrc | grep password | awk '{print $2}')
WANDB_PROJECT=raev3
WANDB_ENTITY=uscgvl
# -------------------------------------------------------------

echo "========================================================"
echo "  raev2 MLS decoder training"
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

# OS-assigned free rendezvous port → avoids EADDRINUSE from a lingering launcher
MASTER_PORT=$(${PYTHON} -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "Rendezvous port: ${MASTER_PORT}"

PYTORCH_ALLOC_CONF=expandable_segments:True ${TORCHRUN} --nproc_per_node=${NGPU} \
    --master-port="${MASTER_PORT}" \
    "${SCRIPT}" \
    --data        "${DATA}" \
    --out-dir     "${OUT_DIR}" \
    --epochs      "${EPOCHS}" \
    --batch-size  "${BATCH}" \
    --precision   "${PRECISION}" \
    --lr          "${LR}" \
    --layers      ${LAYERS} \
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
