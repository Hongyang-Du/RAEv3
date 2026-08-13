#!/usr/bin/env bash
# Sequential 16-epoch Stage-1 ViT-XL decoders on 8xA100, one after another:
#   1) FROZEN Stage-0 MAE  denoise-AE k23 d512  ->  stage1-decoder-mae-denoise-k23-d512
#   2) FROZEN Stage-0 JEPA denoise-AE k23 d512  ->  stage1-decoder-jepa-denoise-k23-d512
# Same recipe as the frozen-fusion 16ep decoders (ViT-XL / lpips 1.0 / disc_start 8);
# only the combine (DenoiseAECombine over the frozen bottleneck E, latent_dim 512) differs.
# Each run auto-resumes from <out_dir>/ckpt_latest.pt.
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

ROOT=/sensei-fs-3/users/hongyangd
PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
LOGDIR="$ROOT/logs/denoise_ae"
mkdir -p "$LOGDIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CKPT_KEEP_RECENT="${CKPT_KEEP_RECENT:-3}"
export WANDB_MODE=offline    # match yesterday's stage-0 denoise runs (offline)

run_one () {   # $1 flavor  $2 config  $3 stage0-combine-ckpt  $4 logfile  $5 rdzv-tag
  local flavor="$1" cfg="$2" combine="$3" log="$4" tag="$5"
  [ -f "$combine" ] || { echo "FATAL: STAGE0_COMBINE missing: $combine"; exit 1; }
  local port; port=$($PY -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
  echo "########## $(date '+%F %T') START $flavor 16ep decoder (${NPROC} GPU) cfg=$cfg ##########"
  STAGE0_COMBINE="$combine" "$TR" \
      --standalone --nnodes=1 --nproc_per_node="$NPROC" \
      --master_port="$port" --rdzv_id="$tag" \
      src/train_decoder.py --config "$cfg" > "$log" 2>&1
  echo "########## $(date '+%F %T') $flavor DONE ##########"
}

CK="$ROOT/ckpt"
run_one mae \
  configs/stage1/decoder/mae-denoise-k23-d512-stage1-decoder-16ep.yaml \
  "$CK/stage0-mae-denoise-k23-d512/combine_for_decoder.pt" \
  "$LOGDIR/decoder_mae_denoise_k23_d512.log"  mae-den-dec-d512

echo "########## $(date '+%F %T') MAE decoder done -> START JEPA decoder ##########"

run_one jepa \
  configs/stage1/decoder/jepa-denoise-k23-d512-stage1-decoder-16ep.yaml \
  "$CK/stage0-jepa-denoise-k23-d512/combine_for_decoder.pt" \
  "$LOGDIR/decoder_jepa_denoise_k23_d512.log"  jepa-den-dec-d512

echo "########## $(date '+%F %T') ALL DONE ##########"
