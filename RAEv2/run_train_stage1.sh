#!/usr/bin/env bash
# ============================================================
#  RAEv2 Stage-1 decoder training — OFFICIAL recipe
#  (src/train_stage1.py, the official entrypoint from the README)
#
#  Encoder:  DINOv3-L K=7 (frozen, layers 11,13,15,17,19,21,23)
#  Decoder:  ViT-XL, trained from scratch
#  Losses:   recon + LPIPS + GAN  (official Stage-1 recipe)
#  Data:     /datasets/imagenet-256  (same partial ImageNet-256 as
#            run_train_attnres.sh; train-only, ~93K images)
#
#  This mirrors the README command:
#      uv run torchrun --nproc_per_node=8 src/train_stage1.py \
#          --config <CONFIG> --results-dir ckpts/stage1 \
#          --precision bf16 --compile --wandb
#  adapted to this box (conda `rae` env, wandb entity/project, local data).
#
#  Run INSIDE the `rae` container, e.g.:
#      docker exec -it rae bash -lc 'cd /workspace/RAEv2 && bash run_train_stage1.sh'
# ============================================================
set -euo pipefail

CONDA_ENV=/opt/conda/envs/rae
TORCHRUN=${CONDA_ENV}/bin/torchrun
SCRIPT=src/train_stage1.py

cd "$(dirname "$(realpath "$0")")"

# ── config ───────────────────────────────────────────────────
NGPU=8
CONFIG=configs/stage1/training/dinov3l-k7-imagenet-local.yaml
RESULTS_DIR=output                       # experiment dir = ${RESULTS_DIR}/${EXPERIMENT_NAME}
PRECISION=bf16                           # official default
COMPILE=true                             # official uses torch.compile

EPOCHS=100                              # total training epochs (overrides config)
CKPT_EVERY=999                           # save a checkpoint every N epochs
VAL_EVERY=500                            # wandb recon-viz cadence (steps); disk recon is per-10-epoch
VAL_IMAGE=assets/samples/sample_1.png    # fixed val images (dir globbed) — same 5 as run_train_attnres.sh

# Required by train_stage1.py (asserted) and the wandb logger.
export EXPERIMENT_NAME=stage1_dinov3l_k7_imagenet
WANDB=true
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=$(grep -A2 'api.wandb.ai' ~/.netrc | grep password | awk '{print $2}')
export WANDB_PROJECT=raev3
export WANDB_ENTITY=uscgvl
# ─────────────────────────────────────────────────────────────

echo "========================================================"
echo "  RAEv2 Stage-1 (official) decoder training"
echo "  GPUs:       ${NGPU}"
echo "  Config:     ${CONFIG}"
echo "  Precision:  ${PRECISION}  |  compile: ${COMPILE}"
echo "  Epochs:     ${EPOCHS}  |  ckpt every ${CKPT_EVERY} ep  |  val every ${VAL_EVERY} steps"
echo "  Experiment: ${EXPERIMENT_NAME}"
echo "  Out dir:    ${RESULTS_DIR}/${EXPERIMENT_NAME}"
echo "  wandb:      ${WANDB} (${WANDB_ENTITY}/${WANDB_PROJECT})"
echo "========================================================"
echo ""

EXTRA_ARGS=""
[[ "${COMPILE}" == "true" ]] && EXTRA_ARGS+=" --compile"
[[ "${WANDB}"   == "true" ]] && EXTRA_ARGS+=" --wandb"

PYTORCH_ALLOC_CONF=expandable_segments:True ${TORCHRUN} --nproc_per_node=${NGPU} "${SCRIPT}" \
    --config              "${CONFIG}" \
    --results-dir         "${RESULTS_DIR}" \
    --precision           "${PRECISION}" \
    --epochs              "${EPOCHS}" \
    --checkpoint-interval "${CKPT_EVERY}" \
    --sample-every        "${VAL_EVERY}" \
    --val-image           "${VAL_IMAGE}" \
    ${EXTRA_ARGS}

echo ""
echo "Done. Outputs in ${RESULTS_DIR}/${EXPERIMENT_NAME}/"
echo "Extract the EMA decoder with: scripts/stage1/extract_decoder.py --use-ema"
