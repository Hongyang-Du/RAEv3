#!/usr/bin/env bash
# ============================================================
#  DiT on the DepthAttnCombine (softmax) fusion latent from the semantic-rent joint
#  fusion+decoder run `rent-k23-depthattn-softmax-ganfusion-2node`.
#    1) compute per-position latent stats of the EMA fusion latent (drop=false)
#    2) train the stage-2 DiT on the stats-normalized fusion latent
#
#    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#        nohup bash run_dit_depthattn_rent_k23.sh > output_full/run_dit_depthattn_rent_k23.log 2>&1 &
# ============================================================
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')
# env: rae_env has torch 2.10; swap to /opt/conda/envs/rae on the cluster if that's yours.
CONDA=${CONDA:-/sensei-fs-3/users/hongyangd/rae_env}; TR=${CONDA}/bin/torchrun; PY=${CONDA}/bin/python
# ImageNet-256 root (must contain train/ ImageFolder or the hf arrow dir the loader expects)
DATA=${DATA:-/mnt/localssd/imagenet-256}

export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
export WANDB_PROJECT=${WANDB_PROJECT:-raev3-full}
export WANDB_ENTITY=${WANDB_ENTITY:-uscgvl}
freeport () { ${PY} -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

CFG=configs/stage2/training/imagenet-dinov3l-depthattn-rent-k23.yaml
CKPT_DIR=/sensei-fs-3/users/hongyangd/ckpt/rent-k23-depthattn-softmax-ganfusion-2node
STATS=${CKPT_DIR}/latent_stats.pt
EXP=dit-depthattn-rent-k23

echo "#####  $(date '+%F %T')  latent stats (fusion, drop=false) -> ${STATS}"
if [[ ! -f "${STATS}" ]]; then
    PYTORCH_ALLOC_CONF=expandable_segments:True ${TR} --nproc_per_node=${NGPU} --master-port=$(freeport) \
        scripts/stage1/compute_latent_stats.py --config "${CFG}" --data-dir "${DATA}" \
        --output-path "${STATS}" --num-samples 200000
fi

echo "#####  $(date '+%F %T')  START ${EXP}"
EXPERIMENT_NAME=${EXP} PYTORCH_ALLOC_CONF=expandable_segments:True ${TR} \
    --nproc_per_node=${NGPU} --master-port=$(freeport) \
    src/train.py --config "${CFG}" --results-dir ckpts_full/stage2 --precision bf16 --wandb
echo "#####  $(date '+%F %T')  DONE ${EXP}"
