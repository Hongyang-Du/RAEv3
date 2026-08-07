#!/usr/bin/env bash
# Single-node 8xA100-80GB local training of the native-k7 (layers [11,13,15,17,19,21,23])
# random-drop decoder at two drop rates, run SEQUENTIALLY in the order 0.9 -> 0.6.
# Same recipe as the p_drop=0.3 anchor (randomdrop-plain-k7-nano-p03-oldnorm.yaml): identical
# layers / lr (2e-4) / GAN (disc_weight 0.75, disc_start 8) / cls_surrogate on / 16 epochs;
# ONLY p_drop (and out_dir/wandb name) differ.
#
# Detached run so closing VSCode / the ssh session does NOT kill training:
#   cd /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2
#   nohup bash run_k7_dropsweep_p09p07p05_8gpu.sh > /sensei-fs-3/users/hongyangd/logs/k7_dropsweep_p09p07p05.log 2>&1 &
#
# Each config auto-resumes from <out_dir>/ckpt_latest.pt, so a preemption just resumes.
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"
REPO="$(pwd)"
ROOT=/sensei-fs-3/users/hongyangd

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
unset VIRTUAL_ENV

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CKPT_KEEP_RECENT="${CKPT_KEEP_RECENT:-2}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-500}"
# wandb: the previously hardcoded key is rotated/invalid (401 "user is not logged in"),
# which crashed rank0 at init. Default to OFFLINE so training is never blocked on auth;
# runs are recorded under ./wandb and can be `wandb sync`-ed later. To log ONLINE, launch
# with a valid key:  WANDB_API_KEY=<key> WANDB_MODE=online bash run_k7_dropsweep_...sh
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_ENTITY=uscgvl
export WANDB_PROJECT=raev3-full

freeport () { ${PY} -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

# The nano/oldnorm configs read data from /mnt/localssd/imagenet-256 (node-local SSD).
# A companion background job stages it from S3 and touches .SYNC_DONE when complete;
# block here until the train arrow is present so torchrun does not crash on a cold dir.
DATA=/mnt/localssd/imagenet-256
echo "##### $(date '+%F %T') waiting for imagenet stage -> $DATA ..."
until [ -f "$DATA/.SYNC_DONE" ] || [ -d "$DATA/imagenet-latents-images" ]; do sleep 30; done
echo "##### $(date '+%F %T') imagenet ready ($(du -sh "$DATA" 2>/dev/null | cut -f1))"

for tag in p09 p06; do
  CFG=configs/stage1/decoder/randomdrop-plain-k7-nano-${tag}-oldnorm.yaml
  PORT=$(freeport)
  echo "##### $(date '+%F %T') START ${tag}  cfg=${CFG}  ngpu=${NGPU}  port=${PORT}"
  "$TR" --nproc_per_node="${NGPU}" --master-port="${PORT}" \
      src/train_decoder.py --config "${CFG}"
  rc=$?
  echo "##### $(date '+%F %T') DONE ${tag} (exit ${rc})"
  [ "$rc" -ne 0 ] && { echo "##### ${tag} failed (exit ${rc}); stopping sweep."; exit "$rc"; }
done
echo "##### $(date '+%F %T') ALL DONE (p09,p06)"
