#!/usr/bin/env bash
# Single-node 8x H100 training for the scaled-up random-drop+cls k23 decoder.
#   nohup bash run_decoder_randomdrop_cls_k23_8gpu.sh > train_decoder_k23.log 2>&1 &
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')

# Use the project .venv (NOT /opt/venv which lacks the project deps).
PY=./.venv/bin/python
TR=./.venv/bin/torchrun
# uv leaves VIRTUAL_ENV=/opt/venv in the shell; unset so it doesn't shadow .venv.
unset VIRTUAL_ENV

export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY="II5ZaCIK0YE3IWnNBl6AJ0Q3z7r_PDUhcyAff6Q1CYJ12EUDlWwltTSm5YGoZwnZWYbZ8PZ40hsmC"
export WANDB_ENTITY=uscgvl
export WANDB_PROJECT=raev3-full

# Pick a free rendezvous port on localhost.
freeport () { ${PY} -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }
PORT=$(freeport)

CFG=configs/stage1/decoder/random-drop-layer-mls-plain-cls-k23.yaml

echo "##### $(date '+%F %T')  TRAIN k23 random-drop+cls on ${NGPU} GPU(s), master_port=${PORT}"
${TR} --nproc_per_node=${NGPU} --master-port=${PORT} \
    src/train_decoder.py --config ${CFG}
echo "##### $(date '+%F %T')  TRAIN done (exit $?)"
