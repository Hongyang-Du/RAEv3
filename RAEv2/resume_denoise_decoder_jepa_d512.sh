#!/usr/bin/env bash
# Resume ONLY the Stage-1 JEPA pixel-decoder (k23,d512) run that was interrupted at ep8.
# Auto-resumes from <out_dir>/ckpt_latest.pt. Mirrors run_stage0_denoise_decoder_d512_seq.sh's
# jepa stage, but skips the (already-done) MAE stage. STAGE0_COMBINE points at the prepped
# frozen-encoder combine ckpt; train_decoder.py then overwrites it with ck["combine"] on resume.
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
REPO="$(pwd)"

for base in /sensei-fs-3 /mnt/remotes/sensei-fs-3; do
  if [ -d "$base/users/hongyangd" ]; then ROOT="$base/users/hongyangd"; break; fi
done
: "${ROOT:?}"

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
[ -x "$PY" ] || { echo "FATAL: env not found at $ROOT/rae_env"; exit 1; }

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CKPT_KEEP_RECENT="${CKPT_KEEP_RECENT:-3}"
# roll ckpt_latest every N steps so a future preemption loses <=N steps, not the whole epoch
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-500}"
export WANDB_MODE=offline    # match the original stage-1 denoise decoder runs (offline)

COMB="$ROOT/ckpt/stage0-jepa-denoise-k23-d512/combine_for_decoder.pt"
CFG="configs/stage1/decoder/jepa-denoise-k23-d512-stage1-decoder-16ep.yaml"
[ -f "$COMB" ] || { echo "FATAL: combine ckpt not found: $COMB"; exit 1; }
[ -d /mnt/localssd/imagenet-256/imagenet-latents-images ] \
  || { echo "FATAL: imagenet-256 not staged on local SSD"; exit 1; }

port=$($PY -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
echo "### $(date '+%F %T') RESUME jepa decoder k23 d512  nproc=$NPROC  cfg=$CFG  combine=$COMB  port=$port"
STAGE0_COMBINE="$COMB" exec "$TR" \
  --standalone --nnodes=1 --nproc_per_node="$NPROC" --master_port="$port" \
  --rdzv_id="denoise-dec-16ep-jepa-d512-resume" \
  src/train_decoder.py --config "$CFG"
