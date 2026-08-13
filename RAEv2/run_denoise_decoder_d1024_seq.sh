#!/usr/bin/env bash
# Sequential Stage-1 pixel-decoder (ViT-XL, 16ep) on the FROZEN Stage-0 denoise-AE
# encoders: MAE (k23,d1024) then JEPA (k23,d1024). Mirrors run_stage0_k23_d1024_seq.sh.
#
# For each flavor:
#   1. convert stage0 `encoder` weights -> a `combine`-keyed ckpt (tools/denoise_encoder_to_combine.py)
#   2. STAGE0_COMBINE=<that ckpt>  ->  train_decoder.py loads + FREEZES the DenoiseAECombine
#   3. decoder trains from scratch, 16ep, same recipe as the fusion cls decoder.
# Auto-resumes each run from <out_dir>/ckpt_latest.pt.
#
# usage:  bash run_denoise_decoder_d1024_seq.sh
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
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-500}"
export WANDB_ENTITY="${WANDB_ENTITY:-uscgvl}"
export WANDB_PROJECT="${WANDB_PROJECT:-raev3-full}"
if [ -f "$HOME/.netrc" ] && grep -q 'api.wandb.ai' "$HOME/.netrc" 2>/dev/null; then
  export WANDB_API_KEY=$(grep -A2 'api.wandb.ai' "$HOME/.netrc" | grep password | awk '{print $2}')
else
  export WANDB_MODE=offline
fi

LOGDIR="$ROOT/logs/denoise_ae"
CBDIR="$ROOT/ckpt/_denoise_combine"          # converted combine-keyed ckpts live here
mkdir -p "$LOGDIR" "$CBDIR"

run_one () {  # $1=flavor  $2=stage0_ckpt_dir  $3=config  $4=master_port
  local flavor="$1" s0dir="$2" cfg="$3" mport="$4"
  local s0="$ROOT/ckpt/$s0dir/ckpt_latest.pt"
  local comb="$CBDIR/${s0dir}-combine.pt"
  [ -f "$s0" ] || { echo "FATAL: stage0 ckpt not found: $s0"; exit 1; }
  echo "---- converting $flavor encoder -> combine ($comb) ----"
  "$PY" tools/denoise_encoder_to_combine.py "$s0" "$comb"
  echo "### $(date '+%F %T') 16ep decoder ($flavor)  nproc=$NPROC  cfg=$cfg  stage0=$comb"
  STAGE0_COMBINE="$comb" "$TR" \
    --standalone --nnodes=1 --nproc_per_node="$NPROC" --master_port="$mport" \
    --rdzv_id="denoise-dec-16ep-$flavor" \
    src/train_decoder.py --config "$cfg"
}

echo "########## $(date '+%F %T') START MAE decoder k23 d1024 (${NPROC} GPU) ##########"
run_one mae  stage0-mae-denoise-k23-d1024 \
  configs/stage1/decoder/mae-denoise-k23-d1024-stage1-decoder-16ep.yaml 29561 \
  > "$LOGDIR/decoder_mae_k23_d1024.log" 2>&1
echo "########## $(date '+%F %T') MAE done -> START JEPA decoder k23 d1024 (${NPROC} GPU) ##########"
run_one jepa stage0-jepa-denoise-k23-d1024 \
  configs/stage1/decoder/jepa-denoise-k23-d1024-stage1-decoder-16ep.yaml 29562 \
  > "$LOGDIR/decoder_jepa_k23_d1024.log" 2>&1
echo "########## $(date '+%F %T') ALL DONE ##########"
