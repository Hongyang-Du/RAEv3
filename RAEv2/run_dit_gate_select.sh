#!/usr/bin/env bash
# ============================================================
#  Stage-2 layer SELECTION run: learnable gate (top-7 straight-through) driven by
#  DiT diffusion loss + frozen-decoder L1 recon anchor. The gate keeps EXACTLY 7
#  layers and learns WHICH 7 (diffusion wants deep, recon keeps it decodable).
#  Watch the "[gate top-7 layers]" log; once stable, fix those 7 and run a normal DiT.
#
#    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup bash run_dit_gate_select.sh \
#        > output_full/run_dit_gate_select.log 2>&1 &
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

CFG=configs/stage2/training/imagenet-dinov3l-k23-learned-gate-recon.yaml
EXP=dit-k23-gate-select-top7-ent

echo "#####  $(date '+%F %T')  START ${EXP}  (top-7 gate + recon anchor)"
EXPERIMENT_NAME=${EXP} PYTORCH_ALLOC_CONF=expandable_segments:True ${TR} \
    --nproc_per_node=${NGPU} --master-port=$(freeport) \
    src/train.py --config "${CFG}" --results-dir ckpts_full/stage2 --precision bf16 --wandb
echo "#####  $(date '+%F %T')  DONE ${EXP}"
