#!/usr/bin/env bash
# Sequential Stage-0 denoise-AE on 8xA100: MAE (k23,d512) then JEPA (k23,d512).
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

export CONDA_ENV=/sensei-fs-3/users/hongyangd/rae_env
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline
LOGDIR=/sensei-fs-3/users/hongyangd/logs/denoise_ae
mkdir -p "$LOGDIR"

echo "########## $(date '+%F %T') START MAE k23 d512 (8 GPU) ##########"
bash run_denoise_ae.sh mae configs/stage0/mae-denoise-k23-d512.yaml \
    > "$LOGDIR/mae_k23_d512.log" 2>&1
echo "########## $(date '+%F %T') MAE done -> START JEPA k23 d512 (8 GPU) ##########"
bash run_denoise_ae.sh jepa configs/stage0/jepa-denoise-k23-d512.yaml \
    > "$LOGDIR/jepa_k23_d512.log" 2>&1
echo "########## $(date '+%F %T') ALL DONE ##########"
