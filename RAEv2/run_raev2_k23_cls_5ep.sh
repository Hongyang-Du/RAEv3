#!/usr/bin/env bash
# RAEv2 K=23 FIXED mean + cls ON, 5ep, GAN@1, lr 8e-4. Control vs random-drop+cls 5ep.
# Train -> 50k eval (native k23 / feed-k7 / feed-L11), same idx convention as the
# random-drop+cls 5ep eval (27.71/0.836/1.69 native).
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup bash run_raev2_k23_cls_5ep.sh > raev2_k23_cls_5ep.log 2>&1 &
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

CFG=configs/stage1/decoder/raev2-k23-cls.yaml
CK=output_full/decoder_raev2_k23_cls_5ep/ckpt_latest.pt

echo "##### $(date '+%F %T')  TRAIN raev2-k23-cls 5ep (fixed mean + cls ON)"
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
run_eval ""                       raev2cls5_k23
run_eval "10,12,14,16,18,20,22"   raev2cls5_k7
run_eval "10"                     raev2cls5_L11

echo "############################################################"
echo "#####  $(date '+%F %T')  RAEV2 K23 CLS 5EP DONE"
echo "############################################################"
