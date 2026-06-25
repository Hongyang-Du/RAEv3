#!/usr/bin/env bash
# ============================================================
#  OmniRAE stage-2 DiT experiments (all cls-on, k=23, DDT-XL, 80ep, 8 GPU).
#  Results -> /sensei-fs-3/users/hongyangd/ckpt/<EXPERIMENT_NAME>/ , ckpt every 10 epochs.
#  wandb -> entity uscgvl, project omnirae.
#
#  exp1  h1-plain : DiT on the PLAIN(xcong) decoder block-0 hidden state h1 (1152-d)
#  exp2  sigreg   : DiT on the SIGReg decoder POST-projector latent (1024-d)
#  exp3  encoder  : DiT on the PLAIN combine latent (1024-d); decode w/ BOTH decoders at eval
#
#  Prereq stats (computed once, already wired into the configs):
#    exp1 -> output_full/h1_decoder_plain_cls_k23_stats.pt              (compute_h1_stats.py)
#    exp2 -> output_full/decoder_..._mlp_sigreg_k23/latent_stats.pt     (compute_latent_stats.py)
#    exp3 -> output_full/decoder_..._plain_k23/latent_stats.pt          (compute_latent_stats.py)
#
#  Usage:  bash run_omnirae_dits.sh <exp1|exp2|exp3>
# ============================================================
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

# --- env auto-detect: prefer the prebuilt .venv; fall back to conda rae env, then build ---
pick_env () {
  if [ -x .venv/bin/python ] && .venv/bin/python -c "import torch" 2>/dev/null; then
    PY=.venv/bin/python; TR=.venv/bin/torchrun; return
  fi
  if [ -x /opt/conda/envs/rae/bin/python ] && /opt/conda/envs/rae/bin/python -c "import torch" 2>/dev/null; then
    PY=/opt/conda/envs/rae/bin/python; TR=/opt/conda/envs/rae/bin/torchrun; return
  fi
  echo "### no working env found; building via setup_rae_env.sh ..."
  bash setup_rae_env.sh
  PY=/opt/conda/envs/rae/bin/python; TR=/opt/conda/envs/rae/bin/torchrun
}
[ -n "${PY:-}" ] && [ -n "${TR:-}" ] || pick_env
NPROC=${NPROC:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
RESULTS_DIR=${RESULTS_DIR:-/sensei-fs-3/users/hongyangd/ckpt}

export WANDB_ENTITY=${WANDB_ENTITY:-uscgvl}
export WANDB_PROJECT=${WANDB_PROJECT:-omnirae}
# export WANDB_KEY=...   # or rely on `wandb login` (already done on this box)

case "${1:-}" in
  exp1) CFG=configs/stage2/training/imagenet-dinov3l-h1decoder-plain-cls-k23.yaml; NAME=omnirae-dit-h1-plain-cls-k23; PORT=29611 ;;
  exp2) CFG=configs/stage2/training/imagenet-dinov3l-sigreg-cls-k23.yaml;          NAME=omnirae-dit-sigreg-cls-k23;   PORT=29612 ;;
  exp3) CFG=configs/stage2/training/imagenet-dinov3l-encoder-cls-k23.yaml;         NAME=omnirae-dit-encoder-cls-k23;  PORT=29613 ;;
  *) echo "usage: bash run_omnirae_dits.sh <exp1|exp2|exp3>"; exit 1 ;;
esac

echo "### $(date '+%F %T')  ${NAME}  cfg=${CFG}  -> ${RESULTS_DIR}/${NAME}"
CUDA_VISIBLE_DEVICES=${GPUS} EXPERIMENT_NAME=${NAME} \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  ${TR} --nproc_per_node=${NPROC} --master-port=${PORT} src/train.py \
  --config "${CFG}" \
  --results-dir "${RESULTS_DIR}" \
  --precision bf16 \
  --wandb
