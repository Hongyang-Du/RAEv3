#!/usr/bin/env bash
# ============================================================
#  Stage-0 denoising-AE feature-fusion launcher (decouples fusion from decoder).
#  Two SEPARATE flavors sharing one bottleneck encoder architecture:
#     mae   -> src/train_mae_denoise.py   (reconstruct full-pool, FIXED target)
#     jepa  -> src/train_jepa_denoise.py  (predict E(full), LIVE target + SIGReg)
#
#  Usage:  bash run_denoise_ae.sh <mae|jepa> <config.yaml> [NGPU]
#  e.g.  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_denoise_ae.sh \
#            mae configs/stage0/mae-denoise-k23-d256.yaml
# ============================================================
set -euo pipefail

FLAVOR=${1:?usage: run_denoise_ae.sh <mae|jepa> config.yaml [NGPU]}
CONFIG=${2:?usage: run_denoise_ae.sh <mae|jepa> config.yaml [NGPU]}
case "${FLAVOR}" in
    mae)  ENTRY=src/train_mae_denoise.py ;;
    jepa) ENTRY=src/train_jepa_denoise.py ;;
    *) echo "FLAVOR must be 'mae' or 'jepa', got '${FLAVOR}'"; exit 1 ;;
esac

CONDA_ENV=${CONDA_ENV:-/opt/conda/envs/rae}    # override on nodes where the env lives elsewhere
TORCHRUN=${CONDA_ENV}/bin/torchrun
PYTHON=${CONDA_ENV}/bin/python
cd "$(dirname "$(realpath "$0")")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NGPU=${3:-$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')}

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=$(grep -A2 'api.wandb.ai' ~/.netrc | grep password | awk '{print $2}')

echo "========================================================"
echo "  Stage-0 denoise-AE (${FLAVOR})  config=${CONFIG}"
echo "  GPUs=${CUDA_VISIBLE_DEVICES}  (NGPU=${NGPU})"
echo "========================================================"

MASTER_PORT=$(${PYTHON} -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "Rendezvous port: ${MASTER_PORT}"

PYTORCH_ALLOC_CONF=expandable_segments:True ${TORCHRUN} --nproc_per_node=${NGPU} \
    --master-port="${MASTER_PORT}" \
    "${ENTRY}" --config "${CONFIG}"
