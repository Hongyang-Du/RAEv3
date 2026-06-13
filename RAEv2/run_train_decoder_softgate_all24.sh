#!/usr/bin/env bash
# ============================================================
#  ALL-LAYER PLAIN SOFTGATE: raev2 + learnable gate, nothing else with the BN projector + global SIGReg
#  (src/train_decoder_mls_softgate_plain.py)
#
#  Encoder:  DINOv3-L (frozen) — ALL 24 blocks (L0..L23)
#  Recipe:   identical to raev2 baseline (L1+LPIPS+GAN from ep2)
#  Combine:   z = sum softmax(gate)_i * layer_i   (the ONLY learnable thing added)
#  SIGReg:   OFF (weight 0, logged only)
#  Decoder:  ViT-XL from scratch; L1+LPIPS+GAN(from ep2)+SIGReg
#
#  Purpose: per-epoch LOO/solo probes map EVERY layer's contribution;
#  final figure: gate trajectory + LOO/solo vs the dropmean twin
# ============================================================
set -euo pipefail

CONDA_ENV=/opt/conda/envs/rae
TORCHRUN=${CONDA_ENV}/bin/torchrun
PYTHON=${CONDA_ENV}/bin/python
SCRIPT=$(dirname "$(realpath "$0")")/src/train_decoder_mls_softgate_plain.py

cd "$(dirname "$(realpath "$0")")"

# -- config ---------------------------------------------------
NGPU=4
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
DATA=/datasets/imagenet-256          # PARTIAL ImageNet (~93K): quick 5-ep gate-collapse test
OUT_DIR=output_full/train_decoder_mls_softgate_all24

EPOCHS=5
BATCH=32
LR=8e-4
PRECISION=bf16
LAYERS=$(seq -s' ' 0 23)        # ALL DINOv3-L blocks
LAYER_DROP=0.0          # pure gate; raev2 recipe unchanged

LPIPS_W=1.0
SIGREG_W=0.0          # no SIGReg constraint (logged only)
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
echo "  ALL-24-layer PLAIN SOFTGATE (no projector / no SIGReg / no dropout)"
echo "  GPUs: ${NGPU}  Batch: ${BATCH}/GPU (global $((BATCH * NGPU)))"
echo "  Epochs: ${EPOCHS}  |  layer_drop ${LAYER_DROP}  |  Layers: ${LAYERS}"
echo "  Output: ${OUT_DIR}"
echo "========================================================"

WANDB_ARGS=""
if [[ "${WANDB}" == "true" ]]; then
    WANDB_ARGS="--wandb --wandb-project ${WANDB_PROJECT} --wandb-entity ${WANDB_ENTITY}"
fi

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
    --layer-drop  "${LAYER_DROP}" \
    --lpips-w     "${LPIPS_W}" \
    --sigreg-w    "${SIGREG_W}" \
    --disc-weight "${DISC_WEIGHT}" \
    --disc-start  "${DISC_START}" \
    --ckpt-every  "${CKPT_EVERY}" \
    --val-every   "${VAL_EVERY}" \
    --log-every   "${LOG_EVERY}" \
    --val-image   "${VAL_IMAGE}" \
    ${WANDB_ARGS}

echo ""
echo "Done. Outputs in ${OUT_DIR}/"
