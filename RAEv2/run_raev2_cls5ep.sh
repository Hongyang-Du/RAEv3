#!/usr/bin/env bash
# RAEv2 fixed-mean + cls baselines, K=23 and K=7, 5ep, GAN@ep2, lr 8e-4, half-batch.
# Same recipe as ours cls-on 5ep (random-drop+cls); isolates fixed-mean vs random-drop.
# Train both -> 50k eval under feed-k7 / feed-k23 / feed-L11 (cross-feed for K=7@k23).
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup bash run_raev2_cls5ep.sh > raev2_cls5ep.log 2>&1 &
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')
CONDA=/opt/conda/envs/rae
TR=${CONDA}/bin/torchrun; PY=${CONDA}/bin/python
export WANDB_API_KEY=$(cat /root/.config/omnirae_wandb_key 2>/dev/null)
export WANDB_ENTITY=uscgvl; export WANDB_PROJECT=raev3-full
freeport(){ ${PY} -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }
DEV=$(echo ${CUDA_VISIBLE_DEVICES} | cut -d, -f1)

K23CFG=configs/stage1/decoder/raev2-k23-cls.yaml
K7CFG=configs/stage1/decoder/raev2-k7-cls.yaml
K23CK=output_full/decoder_raev2_k23_cls_5ep/ckpt_latest.pt
K7CK=output_full/decoder_raev2_k7_cls_5ep/ckpt_latest.pt

train(){ echo "##### $(date '+%F %T') TRAIN $1";
  PYTORCH_ALLOC_CONF=expandable_segments:True ${TR} --nproc_per_node=${NGPU} --master-port=$(freeport) \
    src/train_decoder.py --config $1 || { echo "##### TRAIN_FAILED $1"; exit 1; }; }
ev(){ echo "### $4"; CUDA_VISIBLE_DEVICES=${DEV} PYTORCH_ALLOC_CONF=expandable_segments:True ${PY} \
    src/eval_recon_subset_rfid.py --config $1 --ckpt $2 ${3:+--idx $3} --num-images 50000 --tag $4 --batch 64; }

train ${K23CFG}
train ${K7CFG}

echo "##### $(date '+%F %T') EVAL K=23 cls decoder"
ev ${K23CFG} ${K23CK} ""                       raev2k23cls_k23
ev ${K23CFG} ${K23CK} "10,12,14,16,18,20,22"   raev2k23cls_k7
ev ${K23CFG} ${K23CK} "10"                     raev2k23cls_L11
echo "##### $(date '+%F %T') EVAL K=7 cls decoder"
ev ${K7CFG}  ${K7CK}  ""                        raev2k7cls_k7
ev ${K7CFG}  ${K7CK}  "0"                       raev2k7cls_L11
ev ${K23CFG} ${K7CK}  ""                        raev2k7cls_k23   # cross-feed 23 layers -> K=7 decoder
echo "############################################################"
echo "#####  $(date '+%F %T')  RAEV2 K23/K7 CLS 5EP TRAIN+EVAL DONE"
echo "############################################################"
