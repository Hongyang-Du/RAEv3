#!/usr/bin/env bash
# ============================================================
#  ABLATION: dropmean + BN projector, SIGReg OFF (layers 1..23, raev2 k23 set)
#  (src/train_decoder_mls_dropmean_bn_sigreg.py with --sigreg-w 0)
#
#  Encoder:  DINOv3-L (frozen), layers 1..23 (same as official raev2 k23)
#  Combine:  per-sample random layer dropout 0.3, renormalized mean
#  Projector: fc -> BN(hidden, B*N tokens) -> GELU -> fc (+skip)  [LeWM recipe]
#  SIGReg:   OFF (weight 0) — isolates the BN projector + dropmean from SIGReg.
#  Decoder:  ViT-XL from scratch; L1+LPIPS+GAN(from ep2)
#
#  Ablation pair: vs dropmean_bn (WITH sigreg 0.02) -> effect of SIGReg;
#  vs dropmean_plain (no projector, no sigreg) -> effect of the BN projector.
#  Per-epoch val PSNR+SSIM on 1000 val images; LOO/solo probes at the final epoch.
# ============================================================
set -euo pipefail

CONDA_ENV=/opt/conda/envs/rae
TORCHRUN=${CONDA_ENV}/bin/torchrun
PYTHON=${CONDA_ENV}/bin/python
SCRIPT=$(dirname "$(realpath "$0")")/src/train_decoder_mls_dropmean_bn_sigreg.py

cd "$(dirname "$(realpath "$0")")"

# -- config ---------------------------------------------------
NGPU=4
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}   # GPUs 4-7 busy with another job
DATA=/datasets/imagenet-256-full
OUT_DIR=output_full/train_decoder_mls_dropmean_bn_nosig_k23

EPOCHS=10
BATCH=32
LR=8e-4
PRECISION=bf16
LAYERS=$(seq -s' ' 1 23)        # 1..23, same as official raev2 k23
LAYER_DROP=0.3

LPIPS_W=1.0
SIGREG_W=0.02                    # ABLATION: SIGReg OFF
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
echo "  ABLATION: dropmean + BN projector, SIGReg OFF (k23 layers 1..23)"
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
