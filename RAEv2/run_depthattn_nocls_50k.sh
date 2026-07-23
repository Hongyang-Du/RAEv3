#!/usr/bin/env bash
# depthattn-nocls 5ep ckpt: PSNR/SSIM/rFID(tf)/rFID(fd) on the full 50k official val
# (RAEv3_oldnorm/RAEv2/data_eval/imagenet-256-val.npz), via eval_recon_subset_rfid_mc.py
# (EMA weights, ImageNet de-norm -- the oldnorm 5ep-sweep convention).
# Same feed defs + tags as the depthattn (cls_surrogate:true) 5ep-sweep 50k eval, for
# direct comparison: k23full (full feed), k7 (idx 0..6), l11 (idx 10), l23 (idx 22).
#
#   nohup bash run_depthattn_nocls_50k.sh > ../../logs/depthattn_nocls_50k_driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"
ROOT=/sensei-fs-3/users/hongyangd
PY="$ROOT/rae_env/bin/python"
export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTHONPATH="$ROOT/nanogen-evals/fd_evaluator:$(pwd)/src"
export FD_EVAL=1

CFG=configs/stage1/decoder/randomdrop-plain-k23-nano-p03-depthattn-nocls-5ep-oldnorm.yaml
OD="$ROOT/ckpt/omni-randomdrop-plain-k23-nano-p0.3-depthattn-nocls-5ep-sweep"
LOGD="$ROOT/logs"
NUM=${NUM:-50000}

# Stage the 6.9GB ckpt to local SSD first -- avoids slow 4x concurrent fs-3 reads
# (same rationale as run_p09_50k.sh / run_repro_plain_50k.sh).
mkdir -p /mnt/localssd/stage/depthattn_nocls
CKPT=/mnt/localssd/stage/depthattn_nocls/ckpt_latest.pt
[[ -f $CKPT ]] || cp "$OD/ckpt_latest.pt" "$CKPT"

declare -A IDX=( [k23full]="" [k7]="--idx 0,1,2,3,4,5,6" [l11]="--idx 10" [l23]="--idx 22" )
gpu=0
for feed in k23full k7 l11 l23; do
  tag="depthattn_nocls_${feed}_50k"
  out="$OD/reconrfid_${tag}.json"
  echo "[gpu $gpu] $tag -> $out"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u src/eval_recon_subset_rfid_mc.py --config "$CFG" \
    --ckpt "$CKPT" ${IDX[$feed]} --num-images "$NUM" --batch 32 \
    --tag "$tag" --out "$out" \
    > "$LOGD/eval_${tag}.log" 2>&1 &
  gpu=$((gpu+1))
done
wait
echo ALL_DONE
