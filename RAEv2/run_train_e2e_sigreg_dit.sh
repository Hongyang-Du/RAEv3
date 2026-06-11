#!/usr/bin/env bash
# ============================================================
#  END-TO-END joint training: projector(SIGReg) + decoder + DiT
#  (src/train_e2e_sigreg_dit.py)
#
#  L_rec(dec(z),GT) + SIGReg(z) + FM(zhat,z) + L_pix(dec(zhat),dec(z))
#  All gradient gates OPEN by default (no stop-grad / no EMA target);
#  collapse alarm = falling "Val PSNR (EMA)" lines.
#  Warm-start: projector+decoder from a SIGReg stage-1 ckpt; DiT optional.
# ============================================================
set -euo pipefail

CONDA_ENV=/opt/conda/envs/rae
TORCHRUN=${CONDA_ENV}/bin/torchrun
PYTHON=${CONDA_ENV}/bin/python
SCRIPT=$(dirname "$(realpath "$0")")/src/train_e2e_sigreg_dit.py

cd "$(dirname "$(realpath "$0")")"

# ── config ───────────────────────────────────────────────────
NGPU=8
DATA=/datasets/imagenet-256-full
OUT_DIR=output_full/train_e2e_sigreg_dit

EPOCHS=10
BATCH=24                # per GPU; e2e holds DiT+decoder+projector -> heavier than either stage
LR_PD=1e-4              # projector+decoder AdamW (warm-started, fine-tune pace)
LR_DIT=2e-4             # DiT gmuon, same as stage-2 baselines
PRECISION=bf16

W_REC=1.0
SIGREG_W=1
W_FM=1.0
W_PIX=0.5
EXTRA_FLAGS=""          # safety valves: --detach-fm-target --detach-xt --detach-pix-target --pix-t-weight

INIT_STAGE1=output_full/train_decoder_mls_nogate_sigreg/ckpt_latest.pt   # or dropmean ckpt
INIT_DIT=""             # optional: ckpts_full/stage2/dit-nogate-k7/checkpoints/ep-XXXXXXX.pt

CKPT_EVERY=2
LOG_EVERY=50
SAMPLE_EVERY=2500
VAL_IMAGE=assets/samples/sample_1.png

WANDB=true
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=$(grep -A2 'api.wandb.ai' ~/.netrc | grep password | awk '{print $2}')
WANDB_PROJECT=raev3-full
WANDB_ENTITY=uscgvl
# ─────────────────────────────────────────────────────────────

echo "========================================================"
echo "  E2E projector(SIGReg)+decoder+DiT  |  GPUs: ${NGPU}  Batch: ${BATCH}/GPU (global $((BATCH * NGPU)))"
echo "  Epochs: ${EPOCHS}  |  lr_pd ${LR_PD}  lr_dit ${LR_DIT}  |  w_fm ${W_FM}  w_pix ${W_PIX}"
echo "  Init stage-1: ${INIT_STAGE1}"
echo "  Output: ${OUT_DIR}"
echo "========================================================"

WANDB_ARGS=""
if [[ "${WANDB}" == "true" ]]; then
    WANDB_ARGS="--wandb --wandb-project ${WANDB_PROJECT} --wandb-entity ${WANDB_ENTITY}"
fi
INIT_DIT_ARGS=""
if [[ -n "${INIT_DIT}" ]]; then
    INIT_DIT_ARGS="--init-dit ${INIT_DIT}"
fi

MASTER_PORT=$(${PYTHON} -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "Rendezvous port: ${MASTER_PORT}"

PYTORCH_ALLOC_CONF=expandable_segments:True ${TORCHRUN} --nproc_per_node=${NGPU} \
    --master-port="${MASTER_PORT}" \
    "${SCRIPT}" \
    --data         "${DATA}" \
    --out-dir      "${OUT_DIR}" \
    --epochs       "${EPOCHS}" \
    --batch-size   "${BATCH}" \
    --precision    "${PRECISION}" \
    --lr-pd        "${LR_PD}" \
    --lr-dit       "${LR_DIT}" \
    --w-rec        "${W_REC}" \
    --sigreg-w     "${SIGREG_W}" \
    --w-fm         "${W_FM}" \
    --w-pix        "${W_PIX}" \
    --init-stage1  "${INIT_STAGE1}" \
    --ckpt-every   "${CKPT_EVERY}" \
    --log-every    "${LOG_EVERY}" \
    --sample-every "${SAMPLE_EVERY}" \
    --val-image    "${VAL_IMAGE}" \
    ${INIT_DIT_ARGS} ${EXTRA_FLAGS} ${WANDB_ARGS}

echo ""
echo "Done. Outputs in ${OUT_DIR}/"
