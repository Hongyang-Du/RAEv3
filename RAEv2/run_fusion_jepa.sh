#!/usr/bin/env bash
# ============================================================
#  Stage-0 JEPA feature-fusion pretraining launcher (decouples fusion from decoder).
#  Usage:  bash run_fusion_jepa.sh <config.yaml> [NGPU]
#  GPUs:   set CUDA_VISIBLE_DEVICES to pick cards (default 0-7).
#
#  e.g.  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_fusion_jepa.sh \
#            configs/stage0/fusion-jepa-depthattn-k23.yaml
# ============================================================
set -euo pipefail

CONFIG=${1:?usage: run_fusion_jepa.sh config.yaml [NGPU]}
CONDA_ENV=${CONDA_ENV:-/opt/conda/envs/rae}    # override on nodes where the env lives elsewhere
TORCHRUN=${CONDA_ENV}/bin/torchrun
PYTHON=${CONDA_ENV}/bin/python
cd "$(dirname "$(realpath "$0")")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NGPU=${2:-$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')}

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=$(grep -A2 'api.wandb.ai' ~/.netrc | grep password | awk '{print $2}')

echo "========================================================"
echo "  Stage-0 fusion-JEPA  config=${CONFIG}"
echo "  GPUs=${CUDA_VISIBLE_DEVICES}  (NGPU=${NGPU})"
echo "========================================================"

MASTER_PORT=$(${PYTHON} -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "Rendezvous port: ${MASTER_PORT}"

PYTORCH_ALLOC_CONF=expandable_segments:True ${TORCHRUN} --nproc_per_node=${NGPU} \
    --master-port="${MASTER_PORT}" \
    src/train_fusion_jepa.py --config "${CONFIG}"
