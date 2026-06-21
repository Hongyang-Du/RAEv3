#!/usr/bin/env bash
# RAEv2 random-drop p=0.3 + cls ON, 16ep, lr 2e-4, disc_start 8, FULL-batch GAN.
# Aligned to official as far as the RAEv2 code allows. Train -> 50k eval (native /
# feed-k7 / feed-L11).
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup bash run_raev2_randomdrop_cls_gan8.sh > raev2_randomdrop_cls_gan8.log 2>&1 &
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')
CONDA=/opt/conda/envs/rae
TR=${CONDA}/bin/torchrun
PY=${CONDA}/bin/python
export WANDB_API_KEY=$(cat /root/.config/omnirae_wandb_key 2>/dev/null)
export WANDB_ENTITY=uscgvl
export WANDB_PROJECT=raev3-full
freeport () { ${PY} -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

CFG=configs/stage1/decoder/random-drop-layer-mls-plain-cls-k23.yaml
CK=output_full/decoder_random_drop_layer_mls_plain_cls_k23_16ep_gan8/ckpt_latest.pt

echo "##### $(date '+%F %T')  TRAIN random-drop+cls 16ep, full-batch GAN, disc_start 8, lr 2e-4"
PYTORCH_ALLOC_CONF=expandable_segments:True ${TR} --nproc_per_node=${NGPU} --master-port=$(freeport) \
    src/train_decoder.py --config ${CFG} || { echo "##### TRAIN_FAILED"; exit 1; }

DEV=$(echo ${CUDA_VISIBLE_DEVICES} | cut -d, -f1)
run_eval () {  # $1=idx (empty=all)  $2=tag
    echo "### ${2}"
    CUDA_VISIBLE_DEVICES=${DEV} PYTORCH_ALLOC_CONF=expandable_segments:True ${PY} \
        src/eval_recon_subset_rfid.py --config ${CFG} --ckpt ${CK} \
        ${1:+--idx ${1}} --num-images 50000 --tag ${2} --batch 64
}
echo "##### $(date '+%F %T')  50k eval"
run_eval ""                       rdropcls_gan8_k23
run_eval "10,12,14,16,18,20,22"   rdropcls_gan8_k7
run_eval "10"                     rdropcls_gan8_L11

echo "############################################################"
echo "#####  $(date '+%F %T')  RAEV2 RANDOM-DROP+CLS GAN8 16EP DONE"
echo "############################################################"
