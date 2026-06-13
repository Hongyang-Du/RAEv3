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

VARIANT=${1:-nodrop}    # nodrop | drop

CONDA_ENV=/opt/conda/envs/rae
TORCHRUN=${CONDA_ENV}/bin/torchrun
PYTHON=${CONDA_ENV}/bin/python
SCRIPT=$(dirname "$(realpath "$0")")/src/train_e2e_sigreg_dit.py

cd "$(dirname "$(realpath "$0")")"

# ── config ───────────────────────────────────────────────────
NGPU=8
DATA=/datasets/imagenet-256-full

# PURE end-to-end: projector + decoder + DiT all trained FROM SCRATCH (no warm-start).
# INIT_STAGE1 empty -> random init for all three. Set it to a stage-1 ckpt to warm-start.
case "${VARIANT}" in
  nodrop)   # plain mean (nogate-style), deterministic FM target
    OUT_DIR=output_full/train_e2e_nodrop
    LAYER_DROP=0.0
    INIT_STAGE1=""
    ;;
  drop)     # dropmean-style random layer dropout 0.3 (stochastic FM target)
    OUT_DIR=output_full/train_e2e_drop
    LAYER_DROP=0.3
    INIT_STAGE1=""
    ;;
  *) echo "unknown variant: ${VARIANT} (use nodrop|drop)"; exit 1 ;;
esac

EPOCHS=10              # from-scratch: decoder+DiT both need more than the 10-ep warm-start budget
BATCH=24                # per GPU; e2e holds DiT+decoder+projector -> heavier than either stage
LR_PD=2e-4             # projector+decoder AdamW (from scratch -> faster than the 1e-4 fine-tune pace)
LR_DIT=2e-4             # DiT gmuon, same as stage-2 baselines
WARMUP_EPOCHS=2
PRECISION=bf16

W_REC=1.0
SIGREG_W=1.0            # gentle regularizer with the UNSCALED global statistic
                        # (~0.30x recon projector-grad); 0.02 here would be ~off
W_FM=1.0
W_PIX=0.0               # 0 = minimal design (recon + SIGReg + latent FM w/ live target;
                        # FM alone already updates DiT AND projector). >0 adds the
                        # pixel-FM branch dec(zhat) vs dec(z): perceptual reweighting,
                        # costs a 2nd decoder forward + 2nd LPIPS per step.
# projector = LeWM-recipe MLP: Linear -> BN(hidden, over B*N tokens) -> GELU -> Linear
# (+ identity skip); Linear weights warm-start from the LN stage-1 ckpt, BN stats fresh
EXTRA_FLAGS=""          # safety valves: --detach-fm-target --detach-xt --detach-pix-target --pix-t-weight

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
echo "  E2E projector(SIGReg)+decoder+DiT  variant=${VARIANT}"
echo "  GPUs: ${NGPU}  Batch: ${BATCH}/GPU (global $((BATCH * NGPU)))"
echo "  Epochs: ${EPOCHS}  |  lr_pd ${LR_PD}  lr_dit ${LR_DIT}  |  w_fm ${W_FM}  w_pix ${W_PIX}  layer_drop ${LAYER_DROP}"
echo "  Init stage-1: ${INIT_STAGE1:-<none, train from scratch>}"
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
INIT_STAGE1_ARGS=""
if [[ -n "${INIT_STAGE1}" ]]; then
    INIT_STAGE1_ARGS="--init-stage1 ${INIT_STAGE1}"
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
    --layer-drop   "${LAYER_DROP}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --ckpt-every   "${CKPT_EVERY}" \
    --log-every    "${LOG_EVERY}" \
    --sample-every "${SAMPLE_EVERY}" \
    --val-image    "${VAL_IMAGE}" \
    ${INIT_STAGE1_ARGS} ${INIT_DIT_ARGS} ${EXTRA_FLAGS} ${WANDB_ARGS}

echo ""
echo "Done. Outputs in ${OUT_DIR}/"
