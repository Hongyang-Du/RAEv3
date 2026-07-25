#!/usr/bin/env bash
# Waits for the currently-running uniformonly depthattn-nocls 5ep training
# (randomdrop-plain-k23-nano-uniformonly-depthattn-nocls-5ep-oldnorm.yaml) to
# finish, then runs the same 50k rFID eval protocol as
# run_depthattn_nocls_dropsweep.sh but only for the k23full and k7 (layers
# 11,13,15,17,19,21,23) feeds -- no l11/l23 singles.
#
#   nohup bash run_uniformonly_eval_after_train.sh > ../../logs/uniformonly_eval_driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"
ROOT=/sensei-fs-3/users/hongyangd
PY="$ROOT/rae_env/bin/python"
LOGD="$ROOT/logs"

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT/nanogen-evals/fd_evaluator:$(pwd)/src"
export FD_EVAL=1

CFG=configs/stage1/decoder/randomdrop-plain-k23-nano-uniformonly-depthattn-nocls-5ep-oldnorm.yaml
OD="$ROOT/ckpt/omni-randomdrop-plain-k23-nano-uniformonly-depthattn-nocls-5ep-sweep"
TAG=depthattn_nocls_uniformonly
TRAIN_PAT="train_decoder.py --config ${CFG}"

echo "##### $(date '+%F %T')  waiting for training (${CFG}) to finish"
while pgrep -f "${TRAIN_PAT}" >/dev/null 2>&1; do
  sleep 30
done
echo "##### $(date '+%F %T')  training process exited"

DONE_EP=0
if [ -f "$OD/ckpt_latest.pt" ]; then
  DONE_EP=$("$PY" -c "
import torch
ck = torch.load('$OD/ckpt_latest.pt', map_location='cpu', weights_only=False)
print(ck.get('epoch', 0))
" 2>/dev/null || echo 0)
fi
if [ "${DONE_EP:-0}" -lt 5 ] 2>/dev/null; then
  echo "FATAL: ckpt_latest.pt epoch=${DONE_EP:-0} < 5 -- training did not complete, skipping eval"
  exit 1
fi
echo "  ckpt at epoch ${DONE_EP}, proceeding to eval"

mkdir -p "/mnt/localssd/stage/uniformonly"
CKPT="/mnt/localssd/stage/uniformonly/ckpt_latest.pt"
cp "$OD/ckpt_latest.pt" "$CKPT"

echo "##### $(date '+%F %T')  50k rFID eval (k23full, k7) start"
declare -A IDX=( [k23full]="" [k7]="--idx 10,12,14,16,18,20,22" )
gpu=0
for feed in k23full k7; do
  tag="${TAG}_${feed}_50k"
  out="$OD/reconrfid_${tag}.json"
  echo "  [gpu $gpu] $tag -> $out"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u src/eval_recon_subset_rfid_mc.py --config "$CFG" \
    --ckpt "$CKPT" ${IDX[$feed]} --num-images 50000 --batch 32 \
    --tag "$tag" --out "$out" \
    > "$LOGD/eval_${tag}.log" 2>&1 &
  gpu=$((gpu+1))
done
wait
rm -f "$CKPT"
echo "##### $(date '+%F %T')  50k rFID eval (k23full, k7) DONE"
