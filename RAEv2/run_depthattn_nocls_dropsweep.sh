#!/usr/bin/env bash
# depthattn-nocls (cls_surrogate:false) drop-rate sweep: p_drop = 0.0..0.9 step 0.1,
# 5 epochs each, SAME recipe/hparams as run_depthattn_nocls_5ep.sh (only p_drop
# differs across configs configs/stage1/decoder/randomdrop-plain-k23-nano-p0X-
# depthattn-nocls-5ep-oldnorm.yaml). p0.3 already trained + 50k-evaled earlier
# (ckpt/omni-randomdrop-plain-k23-nano-p0.3-depthattn-nocls-5ep-sweep) -> reused,
# not retrained.
#
# For each leg: train to epoch 5 (resume-safe, skips if already done), stage ckpt
# to local SSD, then 50k rFID eval (k23full/k7/l11/l23) on 4 GPUs, same protocol
# as run_depthattn_nocls_50k.sh. One leg at a time (single 8-GPU node) -- a single
# leg's failure is logged and the sweep continues to the next p_drop.
#
#   nohup bash run_depthattn_nocls_dropsweep.sh > ../../logs/depthattn_nocls_dropsweep_driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"
ROOT=/sensei-fs-3/users/hongyangd
PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
[ -x "$PY" ] || { echo "FATAL: portable env not found at $ROOT/rae_env"; exit 1; }

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')
export WANDB_ENTITY="${WANDB_ENTITY:-uscgvl}"
export WANDB_PROJECT="${WANDB_PROJECT:-raev3-full}"
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline
export PYTHONPATH="$ROOT/nanogen-evals/fd_evaluator:$(pwd)/src"
export FD_EVAL=1

mkdir -p "$ROOT/logs"
LOGD="$ROOT/logs"

freeport () { "$PY" -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

# p_str (config-file tag, no dot) -> p_dot (dir/tag with dot)
declare -A PSTR=( [0.0]=00 [0.1]=01 [0.2]=02 [0.3]=03 [0.4]=04 [0.5]=05 [0.6]=06 [0.7]=07 [0.8]=08 [0.9]=09 )
PDROPS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for PDOT in "${PDROPS[@]}"; do
  PSTR_V="${PSTR[$PDOT]}"
  CFG="configs/stage1/decoder/randomdrop-plain-k23-nano-p${PSTR_V}-depthattn-nocls-5ep-oldnorm.yaml"
  OD="$ROOT/ckpt/omni-randomdrop-plain-k23-nano-p${PDOT}-depthattn-nocls-5ep-sweep"
  TAG="depthattn_nocls_p${PSTR_V}"

  echo "##### $(date '+%F %T')  ===== p_drop=${PDOT}  cfg=${CFG} ====="

  # p0.3 was already trained + 50k-evaled earlier under the old (untagged) file
  # names -- reuse it wholesale rather than retraining or re-running eval.
  if [ "$PDOT" = "0.3" ] && [ -f "$OD/reconrfid_depthattn_nocls_l23_50k.json" ]; then
    echo "  p_drop=0.3: already trained + 50k-evaled earlier, reusing as-is"
    continue
  fi

  # ---- train (resume-safe: skip if already at epoch 5) ----
  DONE_EP=0
  if [ -f "$OD/ckpt_latest.pt" ]; then
    DONE_EP=$("$PY" -c "
import torch
ck = torch.load('$OD/ckpt_latest.pt', map_location='cpu', weights_only=False)
print(ck.get('epoch', 0))
" 2>/dev/null || echo 0)
  fi
  if [ "${DONE_EP:-0}" -ge 5 ] 2>/dev/null; then
    echo "  p_drop=${PDOT}: ckpt already at epoch ${DONE_EP} >= 5, skipping training"
  else
    echo "  p_drop=${PDOT}: training from epoch ${DONE_EP:-0}"
    PORT=$(freeport)
    set +e
    "$TR" --nproc_per_node="${NGPU}" --master-port="${PORT}" \
        src/train_decoder.py --config "${CFG}" 2>&1 | tee -a "$LOGD/dropsweep_train_p${PSTR_V}.log"
    rc=${PIPESTATUS[0]}
    set -e
    echo "##### $(date '+%F %T')  TRAIN p_drop=${PDOT} done (exit ${rc})"
    if [ "$rc" -ne 0 ]; then
      echo "ERROR: p_drop=${PDOT} training failed (exit ${rc}) -- skipping eval, continuing sweep"
      continue
    fi
  fi

  # ---- 50k rFID eval (k23full/k7/l11/l23), ckpt staged to local SSD ----
  if [ -f "$OD/reconrfid_${TAG}_l23_50k.json" ] && [ -f "$OD/reconrfid_${TAG}_k23full_50k.json" ]; then
    echo "  p_drop=${PDOT}: 50k eval already present, skipping"
    continue
  fi
  echo "##### $(date '+%F %T')  50k eval p_drop=${PDOT}"
  mkdir -p "/mnt/localssd/stage/dropsweep_p${PSTR_V}"
  CKPT="/mnt/localssd/stage/dropsweep_p${PSTR_V}/ckpt_latest.pt"
  [[ -f $CKPT ]] || cp "$OD/ckpt_latest.pt" "$CKPT"

  declare -A IDX=( [k23full]="" [k7]="--idx 0,1,2,3,4,5,6" [l11]="--idx 10" [l23]="--idx 22" )
  gpu=0
  for feed in k23full k7 l11 l23; do
    tag="${TAG}_${feed}_50k"
    out="$OD/reconrfid_${tag}.json"
    echo "  [gpu $gpu] $tag -> $out"
    set +e
    CUDA_VISIBLE_DEVICES=$gpu "$PY" -u src/eval_recon_subset_rfid_mc.py --config "$CFG" \
      --ckpt "$CKPT" ${IDX[$feed]} --num-images 50000 --batch 32 \
      --tag "$tag" --out "$out" \
      > "$LOGD/eval_${tag}.log" 2>&1 &
    set -e
    gpu=$((gpu+1))
  done
  wait
  rm -f "$CKPT"
  echo "##### $(date '+%F %T')  50k eval p_drop=${PDOT} DONE"
done

echo "##### $(date '+%F %T')  DROPSWEEP ALL DONE"
