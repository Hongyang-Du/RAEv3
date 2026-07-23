#!/usr/bin/env bash
# Single-node 8xA100: Variant B (depth-attn fusion) 5-epoch smoke run WITHOUT
# cls_surrogate (ablates the additive L23-token-mean term out of DepthAttnCombine).
# Same recipe/budget as run_variant_sweep_5ep_8gpu.sh's depthattn leg, just the one
# variant, so it doesn't touch the shared anchor/maskcond sweep in progress.
# Trains, then runs the same k23full/k7/l11 recon eval as run_variant_sweep_5ep_eval.sh
# for direct comparison against the depthattn (cls_surrogate:true) 5ep ckpt.
#
#   nohup bash run_depthattn_nocls_5ep.sh > logs/depthattn_nocls_5ep_driver.log 2>&1 &
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

mkdir -p "$ROOT/logs"

CFG=configs/stage1/decoder/randomdrop-plain-k23-nano-p03-depthattn-nocls-5ep-oldnorm.yaml
OD="$ROOT/ckpt/omni-randomdrop-plain-k23-nano-p0.3-depthattn-nocls-5ep-sweep"
LOG="$ROOT/logs/variant-sweep-5ep-depthattn-nocls.log"

freeport () { "$PY" -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

echo "##### $(date '+%F %T')  TRAIN depthattn-nocls  cfg=${CFG}  gpus=${NGPU}"
DONE_EP=0
if [ -f "$OD/ckpt_latest.pt" ]; then
  DONE_EP=$("$PY" -c "
import torch
ck = torch.load('$OD/ckpt_latest.pt', map_location='cpu', weights_only=False)
print(ck.get('epoch', 0))
" 2>/dev/null || echo 0)
fi

if [ "${DONE_EP:-0}" -ge 5 ] 2>/dev/null; then
  echo "  depthattn-nocls: ckpt already at epoch ${DONE_EP} >= 5, skipping training"
else
  echo "  depthattn-nocls: training from epoch ${DONE_EP:-0}"
  PORT=$(freeport)
  set +e
  "$TR" --nproc_per_node="${NGPU}" --master-port="${PORT}" \
      src/train_decoder.py --config "${CFG}" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  echo "##### $(date '+%F %T')  TRAIN depthattn-nocls done (exit ${rc})"
  if [ "$rc" -ne 0 ]; then
    echo "FATAL: depthattn-nocls training failed (exit ${rc})"
    exit 1
  fi
fi

CKPT="$OD/ckpt_latest.pt"
echo "##### $(date '+%F %T')  50k-style recon eval (k23full/k7/l11)"
"$PY" src/eval_recon_subset.py --config "${CFG}" --ckpt "${CKPT}" \
    --tag depthattn_nocls_k23full
"$PY" src/eval_recon_subset.py --config "${CFG}" --ckpt "${CKPT}" \
    --idx 0,1,2,3,4,5,6 --tag depthattn_nocls_k7
"$PY" src/eval_recon_subset.py --config "${CFG}" --ckpt "${CKPT}" \
    --idx 10 --tag depthattn_nocls_l11

echo "##### $(date '+%F %T')  DEPTHATTN-NOCLS 5EP DONE"
