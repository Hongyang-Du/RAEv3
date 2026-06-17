#!/usr/bin/env bash
# ============================================================
#  Stage-2 DiT comparison: SIGReg-gaussianized vs plain k=23 Random-Drop latent.
#  For each variant: compute latent stats (drop-ON distribution) if missing, then
#  train DDT-XL 10 epochs on the per-step random-dropped latent.
#
#    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup bash run_dit_compare.sh \
#        > output_full/run_dit_compare.log 2>&1 &
# ============================================================
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')
CONDA=/opt/conda/envs/rae
TR=${CONDA}/bin/torchrun
PY=${CONDA}/bin/python
DATA=/datasets/imagenet-256-full

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=$(grep -A2 'api.wandb.ai' ~/.netrc | grep password | awk '{print $2}')
export WANDB_PROJECT=raev3-full
export WANDB_ENTITY=uscgvl

freeport () { ${PY} -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

run_one () {
    local CFG=$1 S1=$2 EXP=$3
    local STATS=${S1}/latent_stats.pt
    echo "############################################################"
    echo "#####  $(date '+%F %T')  START ${EXP}"
    echo "############################################################"
    if [[ ! -f "${STATS}" ]]; then
        echo "--- latent stats (drop-ON) -> ${STATS} ---"
        PYTORCH_ALLOC_CONF=expandable_segments:True ${TR} --nproc_per_node=${NGPU} --master-port=$(freeport) \
            scripts/stage1/compute_latent_stats.py --config "${CFG}" --data-dir "${DATA}" \
            --output-path "${STATS}" --num-samples 250000
    else
        echo "--- latent stats present: ${STATS} ---"
    fi
    echo "--- DiT training ${EXP} (10 epochs) ---"
    EXPERIMENT_NAME=${EXP} PYTORCH_ALLOC_CONF=expandable_segments:True ${TR} --nproc_per_node=${NGPU} --master-port=$(freeport) \
        src/train.py --config "${CFG}" --results-dir ckpts_full/stage2 --precision bf16 --wandb
    echo "#####  $(date '+%F %T')  DONE ${EXP}  (ckpts: ckpts_full/stage2/${EXP})"
}

run_one configs/stage2/training/imagenet-dinov3l-k23-drop-sigreg.yaml \
        output_full/decoder_random_drop_layer_mls_mlp_sigreg_k23 dit-k23-drop-sigreg
run_one configs/stage2/training/imagenet-dinov3l-k23-drop-plain.yaml \
        output_full/decoder_random_drop_layer_mls_plain_k23 dit-k23-drop-plain

echo "############################################################"
echo "#####  $(date '+%F %T')  ALL DiT DONE"
echo "#####  FID:  python src/eval_fid_dit.py ... (sigreg vs plain)"
echo "############################################################"
