#!/usr/bin/env bash
# ============================================================
#  RAEv2 Width-sweep overfitting experiment
#  Encoder: DINOv3-L (C=1024)  Transport: x-prediction
#  One GPU per width, all widths run in parallel.
# ============================================================
set -euo pipefail

CONDA_ENV=/home/colligo/miniconda3/envs/rae
SEED=42

# --- width sweep (around C=1024) ---
LATENT_DIM=1024
HIDDEN_SIZES="128 256 512 1024 1152 1440"   # same widths as RAEv1 for direct comparison
DEPTH=12
LAUNCH_DELAY=10

# --- training (must match RAEv1 for fair comparison) ---
NUM_STEPS=1000
LR=5e-4
WARMUP_STEPS=100
LOG_INTERVAL=100

# --- validation ---
VAL_EVERY=100
N_VAL_SAMPLES=8
N_FID_SAMPLES=0

# --- output ---
OUTPUT_DIR=output/overfit_results_v2

# --- wandb ---
WANDB=true
WANDB_PROJECT=rae
WANDB_ENTITY=hongyangd

# --- image (same as RAEv1) ---
RAE_CONFIG=configs/stage2/training/imagenet-dinov3l-k7.yaml
IMAGE=../RAE/assets/parrot.png

# ─────────────────────────────────────────────────────────────

PYTHON=${CONDA_ENV}/bin/python
SCRIPT=$(dirname "$(realpath "$0")")/src/overfit_single_image.py

cd "$(dirname "$(realpath "$0")")"

echo "======================================================"
echo "  RAEv2 §4.1 width-sweep  (DINOv3-L, C=${LATENT_DIM})"
echo "  widths : ${HIDDEN_SIZES}"
echo "  steps  : ${NUM_STEPS}   lr : ${LR}   (1 GPU per width)"
echo "  output : ${OUTPUT_DIR}"
echo "  wandb  : ${WANDB} (${WANDB_ENTITY}/${WANDB_PROJECT})"
echo "======================================================"

if [[ "${WANDB}" == "true" && -z "${WANDB_API_KEY:-}" ]]; then
    ${CONDA_ENV}/bin/wandb login
fi

EXTRA_ARGS="--rae-config ${RAE_CONFIG} --image ${IMAGE}"
if [[ "${WANDB}" == "true" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --wandb --wandb-project ${WANDB_PROJECT} --wandb-entity ${WANDB_ENTITY}"
fi

# shellcheck disable=SC2086
${PYTHON} "${SCRIPT}" \
    --sweep \
    --latent-dim    "${LATENT_DIM}" \
    --hidden-sizes  ${HIDDEN_SIZES} \
    --depth         "${DEPTH}" \
    --num-steps     "${NUM_STEPS}" \
    --lr            "${LR}" \
    --warmup-steps  "${WARMUP_STEPS}" \
    --log-interval  "${LOG_INTERVAL}" \
    --val-every     "${VAL_EVERY}" \
    --n-val-samples "${N_VAL_SAMPLES}" \
    --n-fid-samples "${N_FID_SAMPLES}" \
    --launch-delay  "${LAUNCH_DELAY}" \
    --seed          "${SEED}" \
    --output-dir    "${OUTPUT_DIR}" \
    ${EXTRA_ARGS}

echo ""
echo "Done. Results in ${OUTPUT_DIR}/"
