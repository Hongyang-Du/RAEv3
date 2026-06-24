#!/usr/bin/env bash
# Continue-train the downloaded xcong plain-k23 decoder (ckpt ep14) to 16 epochs on 8xH100.
# The ckpt is already placed at out_dir/ckpt_latest.pt, so train_decoder.py auto-resumes
# (epoch=14, step=70056) and the scheduler (last_epoch=70056) continues its LR curve.
# global_batch = 32 x 8 = 256, matching the original 32-GPU run -> 5004 steps/epoch.
#
#   nohup bash run_continue_xcong_16ep.sh > continue_xcong_16ep.log 2>&1 &
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')
PY=./.venv/bin/python
TR=./.venv/bin/torchrun
unset VIRTUAL_ENV

export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/mnt/localssd/.cache/torch
export HF_HOME=/mnt/localssd/.cache/huggingface
export HF_TOKEN="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"
export WANDB_API_KEY="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"
export WANDB_ENTITY=uscgvl
export WANDB_PROJECT=raev3-full

freeport () { ${PY} -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }
CFG=configs/stage1/decoder/continue-xcong-plain-cls-k23-16ep.yaml

echo "##### $(date '+%F %T')  CONTINUE xcong plain-k23 ep14 -> ep16 on ${NGPU} GPUs"
${TR} --nproc_per_node=${NGPU} --master-port=$(freeport) \
    src/train_decoder.py --config ${CFG}
echo "##### $(date '+%F %T')  CONTINUE done (exit $?)"
