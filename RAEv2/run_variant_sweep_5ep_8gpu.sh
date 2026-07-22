#!/usr/bin/env bash
# Single-node 8xA100 sweep: anchor vs Variant A (mask conditioning) vs Variant B
# (depth-attn fusion), all trained FROM SCRATCH at the SAME short budget (5 epochs,
# 2 GAN-free + 3 GAN epochs -- disc_start=2) so the comparison isolates the
# architecture change from the training budget. See RAEv3_oldnorm/README.md for the
# full-budget (16ep) version of this same 3-way ablation.
#
# Trains the 3 configs SEQUENTIALLY (each uses all 8 GPUs), then runs the k23/k7/l11
# eval sweep (run_variant_sweep_5ep_eval.sh) and compiles the comparison table
# (compile_variant_sweep_5ep_csv.py).
#
#   nohup bash run_variant_sweep_5ep_8gpu.sh > logs_5ep_sweep_driver.log 2>&1 &
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
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline   # no key in env -> log locally, don't block on login

mkdir -p "$ROOT/logs"

NAMES=(anchor maskcond depthattn)
declare -A CFG=(
  [anchor]="configs/stage1/decoder/randomdrop-plain-k23-nano-p03-5ep-oldnorm.yaml"
  [maskcond]="configs/stage1/decoder/randomdrop-plain-k23-nano-p03-maskcond-5ep-oldnorm.yaml"
  [depthattn]="configs/stage1/decoder/randomdrop-plain-k23-nano-p03-depthattn-5ep-oldnorm.yaml"
)
declare -A OUTDIR=(
  [anchor]="$ROOT/ckpt/omni-randomdrop-plain-k23-nano-p0.3-5ep-sweep"
  [maskcond]="$ROOT/ckpt/omni-randomdrop-plain-k23-nano-p0.3-maskcond-5ep-sweep"
  [depthattn]="$ROOT/ckpt/omni-randomdrop-plain-k23-nano-p0.3-depthattn-5ep-sweep"
)

freeport () { "$PY" -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

for NAME in "${NAMES[@]}"; do
  CFGFILE="${CFG[$NAME]}"
  OD="${OUTDIR[$NAME]}"
  LOG="$ROOT/logs/variant-sweep-5ep-${NAME}.log"
  echo "##### $(date '+%F %T')  TRAIN ${NAME}  cfg=${CFGFILE}  gpus=${NGPU}"

  if [ -f "$OD/ckpt_latest.pt" ]; then
    DONE_EP=$("$PY" -c "
import torch
ck = torch.load('$OD/ckpt_latest.pt', map_location='cpu', weights_only=False)
print(ck.get('epoch', 0))
" 2>/dev/null)
    if [ -n "$DONE_EP" ] && [ "$DONE_EP" -ge 5 ] 2>/dev/null; then
      echo "  ${NAME}: ckpt already at epoch ${DONE_EP} >= 5, skipping training"
      continue
    fi
    echo "  ${NAME}: resuming from epoch ${DONE_EP:-0}"
  fi

  PORT=$(freeport)
  set +e
  "$TR" --nproc_per_node="${NGPU}" --master-port="${PORT}" \
      src/train_decoder.py --config "${CFGFILE}" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  echo "##### $(date '+%F %T')  TRAIN ${NAME} done (exit ${rc})"
  if [ "$rc" -ne 0 ]; then
    echo "FATAL: ${NAME} training failed (exit ${rc}), aborting sweep"
    exit 1
  fi
done

echo "##### $(date '+%F %T')  all 3 trainings done -> eval sweep (k23/k7/l11)"
bash run_variant_sweep_5ep_eval.sh

echo "##### $(date '+%F %T')  compiling comparison table"
"$PY" compile_variant_sweep_5ep_csv.py

echo "##### $(date '+%F %T')  SWEEP DONE"
