#!/usr/bin/env bash
# ============================================================
#  L11-ONLY decoder training control  (src/train_decoder_mls.py)
#  Encoder:  DINOv3-L (frozen, ONLY layer 11 -- the shallowest of the K7 set)
#  Combine:  raev2 dinov3mls combine on a single layer
#            (mean over {L11} = L11, + L11 token-mean CLS surrogate)
#  Decoder:  ViT-XL (trained from scratch) -- the ONLY trainable module
#
#  Identical recipe to run_train_decoder_mls.sh (the raev2 K7 baseline),
#  ONLY LAYERS and OUT_DIR differ. Control for: does the decoder/DiT
#  actually need the deeper layers, or is the shallowest layer enough?
# ============================================================
set -euo pipefail

CONDA_ENV=/opt/conda/envs/rae
TORCHRUN=${CONDA_ENV}/bin/torchrun
PYTHON=${CONDA_ENV}/bin/python
SCRIPT=$(dirname "$(realpath "$0")")/src/train_decoder_mls.py

cd "$(dirname "$(realpath "$0")")"

# -- config ---------------------------------------------------
NGPU=8
DATA=/datasets/imagenet-256-full
OUT_DIR=output_full/train_decoder_mls_l11

EPOCHS=5
BATCH=32                 # per GPU; global = BATCH x NGPU
LR=8e-4
PRECISION=bf16
LAYERS="11"              # <-- the ONLY change vs raev2: shallowest layer only

LPIPS_W=1.0
DISC_WEIGHT=0.75
DISC_START=1

CKPT_EVERY=1
VAL_EVERY=500
LOG_EVERY=50
VAL_IMAGE=assets/samples/sample_1.png

WANDB=true
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=$(grep -A2 'api.wandb.ai' ~/.netrc | grep password | awk '{print $2}')
WANDB_PROJECT=raev3-full
WANDB_ENTITY=uscgvl
# -------------------------------------------------------------

echo "========================================================"
echo "  L11-only decoder training (raev2 recipe, single layer)"
echo "  GPUs:      ${NGPU}"
echo "  Batch:     ${BATCH}/GPU  (global: $((BATCH * NGPU)))"
echo "  Epochs:    ${EPOCHS}  |  Precision: ${PRECISION}"
echo "  Layers:    ${LAYERS}"
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
