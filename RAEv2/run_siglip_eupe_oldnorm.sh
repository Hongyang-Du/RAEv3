#!/usr/bin/env bash
# ============================================================
#  siglip2-l-k23 / eupe-b-k11 sweep in the RAEv3_oldnorm repo (OLD decoder-output
#  convention: the decoder predicts NORMALIZED pixels, decode de-normalizes).
#
#    bash run_siglip_eupe_oldnorm.sh dec <tag>    # stage-1 decoder, 16 ep
#    bash run_siglip_eupe_oldnorm.sh dit <tag>    # stage-2 DiT-B, 40 ep (needs the p0.3 decoder)
#      tags: eupe_p0 eupe_p03 eupe_p06 eupe_p09 siglip_p0 siglip_p03 siglip_p06 siglip_p09
#
#  One 8-GPU node per invocation. Both stages auto-resume, so a preempted run is
#  restarted with the exact same command:
#    dec -> train_decoder.py resumes from <out_dir>/ckpt_latest.pt
#    dit -> train.py resumes via find_resume_checkpoint(<results-dir>/<EXPERIMENT_NAME>)
#
#  ORDER: the DiT configs point their sample-viz decoder at the p0.3 decoder, so run
#  `dec eupe_p03` / `dec siglip_p03` before any DiT of that encoder (the dit path
#  hard-fails early if that ckpt_ep016.pt is missing).
#
#  These write to *-oldnorm ckpt dirs and never touch the RAEv3-trained ckpts of the
#  same name (those use the [0,1] decoder-output convention and are NOT interchangeable).
#
#  Env: NGPU, DATA, WANDB=1, FG=1 (foreground), NUM_STATS_SAMPLES
# ============================================================
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

STAGE=${1:-}; TAG=${2:-}
case "${TAG}" in
  eupe_p0)    ENC=eupe-b-k11;   PCODE=p0  ; PVAL=0.0 ;;
  eupe_p03)   ENC=eupe-b-k11;   PCODE=p03 ; PVAL=0.3 ;;
  eupe_p06)   ENC=eupe-b-k11;   PCODE=p06 ; PVAL=0.6 ;;
  eupe_p09)   ENC=eupe-b-k11;   PCODE=p09 ; PVAL=0.9 ;;
  siglip_p0)  ENC=siglip-l-k23; PCODE=p0  ; PVAL=0.0 ;;
  siglip_p03) ENC=siglip-l-k23; PCODE=p03 ; PVAL=0.3 ;;
  siglip_p06) ENC=siglip-l-k23; PCODE=p06 ; PVAL=0.6 ;;
  siglip_p09) ENC=siglip-l-k23; PCODE=p09 ; PVAL=0.9 ;;
  *) echo "usage: bash $(basename "$0") <dec|dit> <eupe|siglip>_<p0|p03|p06|p09>"; exit 1 ;;
esac
case "${STAGE}" in dec|dit) ;; *) echo "usage: bash $(basename "$0") <dec|dit> <tag>"; exit 1 ;; esac

export ROOT=/sensei-fs-3/users/hongyangd
export CONDA=${CONDA:-${ROOT}/rae_env}
export TR=${CONDA}/bin/torchrun PY=${CONDA}/bin/python
export DATA=${DATA:-/mnt/localssd/imagenet-256}
export CKPT_ROOT=${ROOT}/ckpt
export STAGE TAG ENC PCODE PVAL
export DEC_CFG=configs/stage1/decoder/random-drop-layer-mls-plain-${ENC}-nano-${PCODE}-oldnorm.yaml
export DIT_CFG=configs/stage2/training/imagenet-${ENC}-encoder-cls-drop-${PCODE}-oldnorm.yaml
export DEC_DIR=${CKPT_ROOT}/omni-randomdrop-plain-${ENC}-nano-p${PVAL}-oldnorm
export VIZ_DEC=${CKPT_ROOT}/omni-randomdrop-plain-${ENC}-nano-p0.3-oldnorm/ckpt_ep016.pt
export DIT_EXP=dit-b-drop-${ENC}-p${PVAL}-oldnorm
export STATS=${CKPT_ROOT}/dit-drop-${ENC}-oldnorm/latent_stats_p${PVAL}.pt

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NGPU=${NGPU:-$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=${ROOT}/.cache/torch          # facebookresearch_EUPE_* hub cache
export HF_HOME=${ROOT}/.cache/huggingface       # google/siglip2-large-patch16-256
export CKPT_EVERY_STEPS=${CKPT_EVERY_STEPS:-2500}   # mid-epoch ckpt_latest (preemption)
export CKPT_KEEP_RECENT=${CKPT_KEEP_RECENT:-2}      # 40ep/interval2 = 20 ckpts -> bound disk
export CKPT_KEEP_EVERY=${CKPT_KEEP_EVERY:-10}
export NUM_STATS_SAMPLES=${NUM_STATS_SAMPLES:-250000}
if [[ "${WANDB:-0}" == "1" ]]; then
  export WANDB_ENTITY=${WANDB_ENTITY:-uscgvl} WANDB_PROJECT=${WANDB_PROJECT:-omnirae}
  export WANDB_FLAG=--wandb
else
  export WANDB_MODE=disabled; export WANDB_FLAG=
fi
LOGDIR=${ROOT}/logs; mkdir -p "${LOGDIR}"
LOG=${LOGDIR}/oldnorm_${STAGE}_${TAG}_$(date +%Y%m%d-%H%M%S).log

freeport () { ${PY} -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

prep () {
  # -- dataset on localssd (idempotent; sync skips what is already there) ------
  mkdir -p "${DATA}"
  aws s3 sync s3://hongyangd-raev2-backup/raev2-data/imagenet-256/ "${DATA}/" \
    || { echo "### FATAL: S3 sync failed (need AWS creds/role on node)"; exit 1; }
  [ -d "${DATA}/imagenet-latents-images" ] \
    || { echo "### FATAL: ${DATA}/imagenet-latents-images missing after sync"; exit 1; }
  mkdir -p data && ln -sfn "${DATA}" data/imagenet-256
  echo "##### staged: $(du -sh "${DATA}" 2>/dev/null | cut -f1)"
  # -- encoders (shared fs; download only if the cache is missing) -------------
  compgen -G "${TORCH_HOME}/hub/facebookresearch_EUPE_*" >/dev/null \
    || ${PY} -c "import torch;torch.hub.load('facebookresearch/EUPE','eupe_vit_b16',pretrained=True)"
  [ -d "${HF_HOME}/hub/models--google--siglip2-large-patch16-256" ] \
    || ${PY} -c "from transformers import SiglipVisionModel as M;M.from_pretrained('google/siglip2-large-patch16-256')"
  echo "##### encoders ok"
}

main () {
  echo "##### $(date '+%F %T')  oldnorm ${STAGE} ${TAG}  enc=${ENC} p=${PVAL} ngpu=${NGPU}"
  prep
  if [[ "${STAGE}" == "dec" ]]; then
    mkdir -p "${DEC_DIR}"
    echo "##### $(date '+%F %T')  START decoder ${TAG} -> ${DEC_DIR}"
    ${TR} --nproc_per_node=${NGPU} --master-port=$(freeport) \
      src/train_decoder.py --config "${DEC_CFG}"
    echo "##### $(date '+%F %T')  DONE decoder ${TAG} rc=$?"
  else
    [ -f "${VIZ_DEC}" ] || { echo "### FATAL: sample-viz decoder missing: ${VIZ_DEC}
###        run  bash $(basename "$0") dec ${ENC%%-*}_p03  first"; exit 1; }
    if [[ ! -f "${STATS}" ]]; then
      mkdir -p "$(dirname "${STATS}")"
      echo "##### $(date '+%F %T')  latent stats -> ${STATS}"
      ${TR} --nproc_per_node=${NGPU} --master-port=$(freeport) \
        scripts/stage1/compute_latent_stats.py --config "${DIT_CFG}" --data-dir "${DATA}" \
        --output-path "${STATS}" --num-samples "${NUM_STATS_SAMPLES}" \
        || { echo "### FATAL: latent stats failed"; exit 1; }
    else
      echo "##### latent stats present: ${STATS}"
    fi
    echo "##### $(date '+%F %T')  START DiT ${DIT_EXP} -> ${CKPT_ROOT}/${DIT_EXP}"
    EXPERIMENT_NAME=${DIT_EXP} ${TR} --nproc_per_node=${NGPU} --master-port=$(freeport) \
      src/train.py --config "${DIT_CFG}" --results-dir "${CKPT_ROOT}" \
      --precision bf16 ${WANDB_FLAG}
    echo "##### $(date '+%F %T')  DONE DiT ${DIT_EXP} rc=$?"
  fi
}

if [[ "${FG:-0}" == "1" ]]; then
  main 2>&1 | tee "${LOG}"
else
  # setsid: survive SIGHUP when the launching session ends (killed a whole round before)
  setsid nohup bash -c "$(declare -f main prep freeport); main" < /dev/null > "${LOG}" 2>&1 &
  disown
  echo "launched oldnorm ${STAGE} ${TAG}  pid=$!  log=${LOG}"
fi
