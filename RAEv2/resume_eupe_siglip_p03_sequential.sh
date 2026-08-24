#!/usr/bin/env bash
# Sequential single-node 8-GPU RESUME of the two p0.3 decoder sweeps, from each
# out_dir's ckpt_latest.pt (train_decoder.py auto-resumes if ckpt_latest.pt exists):
#   1) eupe-b-k11  p0.3  (resumes from epoch 10 -> 16)
#   2) siglip-l-k23 p0.3 (resumes from epoch  7 -> 16)   [only after (1) exits 0]
# Matches how the original sweep ran: rae_env torchrun, cwd = RAEv3/RAEv2, shared
# torch/HF caches (EUPE + siglip2 encoders already cached there).
#
#   nohup bash resume_eupe_siglip_p03_sequential.sh > \
#       /sensei-fs-3/users/hongyangd/logs/resume_eupe_siglip_p03_seq.log 2>&1 &
set -uo pipefail
cd /sensei-fs-3/users/hongyangd/RAEv3/RAEv2

TR=/sensei-fs-3/users/hongyangd/rae_env/bin/torchrun
PY=/sensei-fs-3/users/hongyangd/rae_env/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')

export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/sensei-fs-3/users/hongyangd/.cache/torch          # has facebookresearch_EUPE_* hub cache
export HF_HOME=/sensei-fs-3/users/hongyangd/.cache/huggingface       # has siglip2 weights
export WANDB_MODE=disabled                                           # configs have no wandb section; keep it off
export CKPT_EVERY_STEPS=2500                                         # persist ckpt_latest mid-epoch (preemption safety)
export LOGDIR=/sensei-fs-3/users/hongyangd/logs
mkdir -p "$LOGDIR"

freeport () { ${PY} -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

run_one () {  # $1=tag  $2=config
  local tag=$1 cfg=$2
  local log="$LOGDIR/resume_${tag}_$(date +%Y%m%d-%H%M%S).log"
  echo "############################################################"
  echo "##### $(date '+%F %T')  RESUME ${tag}  cfg=${cfg}  ngpu=${NGPU}  log=${log}"
  echo "############################################################"
  ${TR} --nproc_per_node=${NGPU} --master-port=$(freeport) \
      src/train_decoder.py --config "${cfg}" 2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  echo "##### $(date '+%F %T')  ${tag} EXIT rc=${rc}"
  return ${rc}
}

run_one eupe_p03  configs/stage1/decoder/random-drop-layer-mls-plain-eupe-b-k11-nano-p03.yaml
rc1=$?
if [ ${rc1} -ne 0 ]; then
  echo "##### eupe_p03 FAILED (rc=${rc1}); still attempting siglip_p03."
fi

run_one siglip_p03 configs/stage1/decoder/random-drop-layer-mls-plain-siglip-l-k23-nano-p03.yaml
rc2=$?

echo "##### $(date '+%F %T')  ALL DONE  eupe_p03 rc=${rc1}  siglip_p03 rc=${rc2}"
