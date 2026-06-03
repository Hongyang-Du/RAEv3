#!/usr/bin/env bash
# ============================================================
#  Width-sweep overfitting experiment (RAE paper §4.1)
#  Reproduces: "Scaling DiT Width to Match Token Dimensionality"
#
#  Each hidden_size is trained sequentially on all NGPU GPUs.
#  At the end, two figures are saved to OUTPUT_DIR/:
#    loss_curves.png  — smoothed loss curves + final-loss bar chart
#    latent_grid.png  — PCA→RGB latent reconstruction per width
#    image_grid.png   — decoded pixel images (if RAE weights are set)
# ============================================================
set -euo pipefail

# ── user config ──────────────────────────────────────────────
CONDA_ENV=/home/colligo/miniconda3/envs/rae
SEED=42

# --- width sweep ---
# One GPU per width; widths run in parallel.
# GPU assignment: GPU 0 → width 0, GPU 1 → width 1, etc.
LATENT_DIM=768               # token dim C (must match encoder)
HIDDEN_SIZES="192 384 576 768 960 1152"
DEPTH=12                     # DiT depth (fixed across sweep)
LAUNCH_DELAY=10              # seconds between subprocess starts

# --- training (batch=1 per GPU) ---
NUM_STEPS=1000
LR=5e-4
WARMUP_STEPS=100
LOG_INTERVAL=100

# --- validation ---
VAL_EVERY=100        # save decoded image + metrics every N steps
N_VAL_SAMPLES=8      # ODE samples per periodic validation
N_FID_SAMPLES=0      # skip FID (too slow for short runs)

# --- output ---
OUTPUT_DIR=output/overfit_results

# --- wandb ---
WANDB=true                   # set to false to disable
WANDB_PROJECT=rae
WANDB_ENTITY=hongyangd

# --- real image (optional) ---
# Set both to use a real image via the pretrained RAE encoder.
# Leave empty to use a random latent (no weights needed).
# With a real image, LPIPS + decoded image panels are also logged to wandb.
# RAE_CONFIG=""
# IMAGE=""
# example:
RAE_CONFIG=configs/stage2/training/ImageNet256/DiTDH-S_DINOv2-B.yaml
IMAGE=assets/parrot.png
# ─────────────────────────────────────────────────────────────

PYTHON=${CONDA_ENV}/bin/python
SCRIPT=$(dirname "$(realpath "$0")")/src/overfit_single_image.py

cd "$(dirname "$(realpath "$0")")"

echo "======================================================"
echo "  RAE §4.1 width-sweep experiment"
echo "  widths : ${HIDDEN_SIZES}"
echo "  latent_dim C = ${LATENT_DIM}  depth = ${DEPTH}"
echo "  steps  : ${NUM_STEPS}   lr : ${LR}   (1 GPU per width)"
echo "  output : ${OUTPUT_DIR}"
echo "  wandb  : ${WANDB} (${WANDB_ENTITY}/${WANDB_PROJECT})"
echo "  delay  : ${LAUNCH_DELAY}s between launches"
echo "======================================================"

# login check
if [[ "${WANDB}" == "true" ]]; then
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "  WANDB_API_KEY not set — running: wandb login"
        ${CONDA_ENV}/bin/wandb login
    fi
fi

EXTRA_ARGS=""
if [[ -n "${RAE_CONFIG}" && -n "${IMAGE}" ]]; then
    EXTRA_ARGS="--rae-config ${RAE_CONFIG} --image ${IMAGE}"
    echo "  mode   : real image (${IMAGE})"
else
    echo "  mode   : random latent (seed=${SEED})"
fi
if [[ "${WANDB}" == "true" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --wandb --wandb-project ${WANDB_PROJECT} --wandb-entity ${WANDB_ENTITY}"
fi
echo ""

# shellcheck disable=SC2086
${PYTHON} "${SCRIPT}" \
    --sweep \
    --latent-dim   "${LATENT_DIM}" \
    --hidden-sizes ${HIDDEN_SIZES} \
    --depth        "${DEPTH}" \
    --num-steps    "${NUM_STEPS}" \
    --lr           "${LR}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --log-interval  "${LOG_INTERVAL}" \
    --val-every     "${VAL_EVERY}" \
    --n-val-samples "${N_VAL_SAMPLES}" \
    --n-fid-samples "${N_FID_SAMPLES}" \
    --launch-delay  "${LAUNCH_DELAY}" \
    --seed         "${SEED}" \
    --output-dir   "${OUTPUT_DIR}" \
    ${EXTRA_ARGS}

echo ""
echo "Done. Results in ${OUTPUT_DIR}/"
