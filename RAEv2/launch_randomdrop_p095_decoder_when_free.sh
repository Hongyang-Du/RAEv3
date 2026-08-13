#!/usr/bin/env bash
# Train the p_drop=0.95 randomdrop-plain-k23-nano decoder (16ep, ViT-XL, 8 GPU), extending
# the omni-randomdrop-plain-k23-nano-p0.X sweep to p0.95. Waits for the GPUs to free (the
# MAE-d512 DiT is using them), then launches. train_decoder.py auto-resumes from
# <out_dir>/ckpt_latest.pt, so re-launching continues where it stopped.
# NOTE: MLSCombine is instantiated + trained inline (no STAGE0_COMBINE / frozen encoder).
set -uo pipefail
cd /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2
REPO="$(pwd)"; ROOT=/sensei-fs-3/users/hongyangd

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
CFG=configs/stage1/decoder/randomdrop-plain-k23-nano-p095-oldnorm.yaml

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CKPT_KEEP_RECENT="${CKPT_KEEP_RECENT:-3}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-500}"
export WANDB_MODE=offline

[ -d /mnt/localssd/imagenet-256/imagenet-latents-images ] \
  || { echo "FATAL: imagenet-256 not staged on local SSD"; exit 1; }

# 1) Wait for the in-flight MAE-d512 DiT job to fully finish. Gating on the job process
#    (not just GPU memory) avoids stealing the GPUs during its latent_stats->train gap or
#    its train.py model-load window, when the GPUs briefly read idle.
echo "### $(date '+%F %T') p095 decoder launcher: waiting for the MAE-d512 DiT job to finish..."
while pgrep -f "run_ablation_dits.sh mae-denoise-d512" >/dev/null \
   || pgrep -f "compute_latent_stats.py .*mae-denoise-k23-d512" >/dev/null \
   || pgrep -f "src/train.py .*mae-denoise-k23-d512" >/dev/null; do
  sleep 60
done

# 2) Belt-and-suspenders: confirm the GPUs are actually idle before grabbing them.
echo "### $(date '+%F %T') MAE-d512 job gone; confirming all $NPROC GPUs idle (<2GB used)..."
while :; do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{c++} END{print c+0}')
  if [ "${busy:-1}" -eq 0 ]; then
    sleep 25
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{c++} END{print c+0}')
    [ "${busy:-1}" -eq 0 ] && break
  fi
  sleep 60
done

port=$($PY -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
echo "### $(date '+%F %T') GPUs free -> START p095 decoder  nproc=$NPROC  cfg=$CFG  port=$port"
exec "$TR" --standalone --nnodes=1 --nproc_per_node="$NPROC" --master_port="$port" \
  --rdzv_id="randomdrop-p095-dec" \
  src/train_decoder.py --config "$CFG"
