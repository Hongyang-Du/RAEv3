#!/usr/bin/env bash
# ============================================================
#  Stage-2 DiT with a LEARNED layer gate ("DiT picks its own layers").
#  The K=23 layer combine uses a trainable softmax gate (init uniform 1/23),
#  learned by the DiT denoising loss. No latent-stats precompute is needed:
#  training standardizes the gate-weighted latent per-batch to unit variance.
#  Watch wandb gate/entropy + gate/L* to see if the gate polarizes.
#
#    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup bash run_dit_learned_gate.sh \
#        > output_full/run_dit_learned_gate.log 2>&1 &
# ============================================================
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')
CONDA=/opt/conda/envs/rae
TR=${CONDA}/bin/torchrun
PY=${CONDA}/bin/python

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=$(grep -A2 'api.wandb.ai' ~/.netrc | grep password | awk '{print $2}')
export WANDB_PROJECT=raev3-full
export WANDB_ENTITY=uscgvl

freeport () { ${PY} -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

CFG=configs/stage2/training/imagenet-dinov3l-k23-learned-gate.yaml
EXP=dit-k23-learned-gate

echo "############################################################"
echo "#####  $(date '+%F %T')  START ${EXP}"
echo "############################################################"
EXPERIMENT_NAME=${EXP} PYTORCH_ALLOC_CONF=expandable_segments:True ${TR} \
    --nproc_per_node=${NGPU} --master-port=$(freeport) \
    src/train.py --config "${CFG}" --results-dir ckpts_full/stage2 --precision bf16 --wandb
echo "#####  $(date '+%F %T')  DONE ${EXP}  (ckpts: ckpts_full/stage2/${EXP})"
