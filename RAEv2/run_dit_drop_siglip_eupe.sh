#!/usr/bin/env bash
# ============================================================
#  Stage-2 DiT-B on the siglip2-l-k23 / eupe-b-k11 combine latent, drop sweep.
#  Mirrors run_dit_depthattn_rent_k23.sh (stats -> train) + the decoder sweep's
#  node prep (localssd staging, shared encoder caches, setsid launch).
#
#    bash run_dit_drop_siglip_eupe.sh <tag>
#      tags: eupe_p0  eupe_p03  eupe_p06  eupe_p09
#            siglip_p0 siglip_p03 siglip_p06 siglip_p09
#
#  One node (8 GPU) per tag. Decoder is sampling-viz only and is shared:
#    eupe   -> p0.3 decoder     siglip -> p0.6 decoder
#  train.py auto-resumes from results-dir via find_resume_checkpoint(), so a
#  preempted run is restarted with the exact same command.
#
#  Env overrides: NGPU, DATA, WANDB (set WANDB=1 to log), FG=1 (run in foreground)
# ============================================================
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

TAG=${1:-}
case "${TAG}" in
  eupe_p0)    CFG=configs/stage2/training/imagenet-eupe-b-k11-encoder-cls-drop-p0.yaml;    ENC=eupe-b-k11;   PVAL=0.0 ;;
  eupe_p03)   CFG=configs/stage2/training/imagenet-eupe-b-k11-encoder-cls-drop-p03.yaml;   ENC=eupe-b-k11;   PVAL=0.3 ;;
  eupe_p06)   CFG=configs/stage2/training/imagenet-eupe-b-k11-encoder-cls-drop-p06.yaml;   ENC=eupe-b-k11;   PVAL=0.6 ;;
  eupe_p09)   CFG=configs/stage2/training/imagenet-eupe-b-k11-encoder-cls-drop-p09.yaml;   ENC=eupe-b-k11;   PVAL=0.9 ;;
  siglip_p0)  CFG=configs/stage2/training/imagenet-siglip-l-k23-encoder-cls-drop-p0.yaml;  ENC=siglip-l-k23; PVAL=0.0 ;;
  siglip_p03) CFG=configs/stage2/training/imagenet-siglip-l-k23-encoder-cls-drop-p03.yaml; ENC=siglip-l-k23; PVAL=0.3 ;;
  siglip_p06) CFG=configs/stage2/training/imagenet-siglip-l-k23-encoder-cls-drop-p06.yaml; ENC=siglip-l-k23; PVAL=0.6 ;;
  siglip_p09) CFG=configs/stage2/training/imagenet-siglip-l-k23-encoder-cls-drop-p09.yaml; ENC=siglip-l-k23; PVAL=0.9 ;;
  *) echo "usage: bash $(basename "$0") <eupe|siglip>_<p0|p03|p06|p09>"; exit 1 ;;
esac

# exported throughout: the background branch re-enters via a fresh `bash -c`
export ROOT=/sensei-fs-3/users/hongyangd
export CONDA=${CONDA:-${ROOT}/rae_env} TR=${CONDA:-${ROOT}/rae_env}/bin/torchrun PY=${CONDA:-${ROOT}/rae_env}/bin/python
export DATA=${DATA:-/mnt/localssd/imagenet-256}
export RESULTS_DIR=${RESULTS_DIR:-${ROOT}/ckpt}
export CFG TAG ENC PVAL
export EXP=dit-b-drop-${ENC}-p${PVAL}
export STATS=${ROOT}/ckpt/dit-drop-${ENC}/latent_stats_p${PVAL}.pt   # must match config's normalization_stat_path
LOGDIR=${ROOT}/logs; mkdir -p "${LOGDIR}"
LOG=${LOGDIR}/dit_${TAG}_$(date +%Y%m%d-%H%M%S).log

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NGPU=${NGPU:-$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=${ROOT}/.cache/torch          # facebookresearch_EUPE_* hub cache
export HF_HOME=${ROOT}/.cache/huggingface       # google/siglip2-large-patch16-256
# ckpt retention: 40ep / checkpoint_interval 2 = 20 ckpts/run, ~6.5G each -- bound the disk
export CKPT_KEEP_RECENT=${CKPT_KEEP_RECENT:-2}
export CKPT_KEEP_EVERY=${CKPT_KEEP_EVERY:-10}
export NUM_STATS_SAMPLES=${NUM_STATS_SAMPLES:-250000}
if [[ "${WANDB:-0}" == "1" ]]; then
  export WANDB_ENTITY=${WANDB_ENTITY:-uscgvl} WANDB_PROJECT=${WANDB_PROJECT:-omnirae}
  export WANDB_FLAG=--wandb
else
  export WANDB_MODE=disabled; export WANDB_FLAG=
fi
freeport () { ${PY} -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

main () {
  echo "##### $(date '+%F %T')  ${EXP}  tag=${TAG}  cfg=${CFG}  ngpu=${NGPU}"

  # -- 1) dataset on localssd (idempotent; sync skips what's already there) ----
  mkdir -p "${DATA}"
  aws s3 sync s3://hongyangd-raev2-backup/raev2-data/imagenet-256/ "${DATA}/" \
    || { echo "### FATAL: S3 sync failed (need AWS creds/role on node)"; exit 1; }
  echo "##### staged: $(du -sh "${DATA}" 2>/dev/null | cut -f1)  files=$(find "${DATA}" -type f | wc -l)"
  [ -d "${DATA}/imagenet-latents-images" ] || { echo "### FATAL: ${DATA}/imagenet-latents-images missing after sync"; exit 1; }
  mkdir -p data && ln -sfn "${DATA}" data/imagenet-256   # loader also resolves the repo-relative path

  # -- 2) encoders (on shared fs; download only if the cache is missing) -------
  compgen -G "${TORCH_HOME}/hub/facebookresearch_EUPE_*" >/dev/null \
    || ${PY} -c "import torch;torch.hub.load('facebookresearch/EUPE','eupe_vit_b16',pretrained=True)"
  [ -d "${HF_HOME}/hub/models--google--siglip2-large-patch16-256" ] \
    || ${PY} -c "from transformers import SiglipVisionModel as M;M.from_pretrained('google/siglip2-large-patch16-256')"
  echo "##### encoders ok"

  # -- 3) latent stats, drop-matched to this config (skipped if present) -------
  DEC=$(${PY} -c "import yaml,sys;print(yaml.safe_load(open('${CFG}'))['stage_1']['params']['stage1_ckpt_path'] or '')")
  [ -z "${DEC}" ] || [ -f "${DEC}" ] || { echo "### FATAL: decoder ckpt missing: ${DEC}"; exit 1; }
  if [[ ! -f "${STATS}" ]]; then
    mkdir -p "$(dirname "${STATS}")"
    echo "##### $(date '+%F %T')  latent stats -> ${STATS}"
    ${TR} --nproc_per_node=${NGPU} --master-port=$(freeport) \
      scripts/stage1/compute_latent_stats.py --config "${CFG}" --data-dir "${DATA}" \
      --output-path "${STATS}" --num-samples "${NUM_STATS_SAMPLES}" \
      || { echo "### FATAL: latent stats failed"; exit 1; }
  else
    echo "##### latent stats present: ${STATS}"
  fi

  # -- 4) DiT-B (83 ep; auto-resumes from ${RESULTS_DIR}/${EXP}) ---------------
  echo "##### $(date '+%F %T')  START ${EXP} -> ${RESULTS_DIR}/${EXP}"
  EXPERIMENT_NAME=${EXP} ${TR} --nproc_per_node=${NGPU} --master-port=$(freeport) \
    src/train.py --config "${CFG}" --results-dir "${RESULTS_DIR}" \
    --precision bf16 ${WANDB_FLAG}
  echo "##### $(date '+%F %T')  DONE ${EXP} rc=$?"
}

if [[ "${FG:-0}" == "1" ]]; then
  main 2>&1 | tee "${LOG}"
else
  # setsid: survive SIGHUP when the launching session ends (killed the last round)
  setsid nohup bash -c "$(declare -f main freeport); main" < /dev/null > "${LOG}" 2>&1 &
  disown
  echo "launched ${EXP}  pid=$!  log=${LOG}"
fi
