#!/usr/bin/env bash
# Re-run the MAE-d512 DiT-B (40ep) that FAILED on 2026-08-12 20:39 (latent_stats hung ~7.5h,
# SIGKILL'd, never trained). Waits for the p0.95 decoder job to finish, then:
#   (1) robustly computes latent_stats.pt (timeout+retry -- the previous run hung here), then
#   (2) hands off to run_ablation_dits.sh mae-denoise-d512, which sees latent_stats present,
#       skips the stats step, and trains 40ep (auto-resumes per EXPERIMENT_NAME).
set -uo pipefail
cd /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2
ROOT=/sensei-fs-3/users/hongyangd
PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

CFG=configs/stage2/training/imagenet-dinov3l-mae-denoise-k23-d512-ablB.yaml
STATS="$ROOT/ckpt/stage1-decoder-mae-denoise-k23-d512/latent_stats.pt"
NGPU=8

# 1) wait for the p0.95 decoder job to finish (gate on its process, not just GPU memory)
echo "### $(date '+%F %T') mae-d512 requeue: waiting for the p0.95 decoder job to finish..."
while pgrep -f "train_decoder.py --config configs/stage1/decoder/randomdrop-plain-k23-nano-p095" >/dev/null; do
  sleep 120
done
echo "### $(date '+%F %T') p0.95 done; confirming GPUs idle..."
while :; do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{c++} END{print c+0}')
  [ "${busy:-1}" -eq 0 ] && { sleep 25; busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{c++} END{print c+0}'); [ "${busy:-1}" -eq 0 ] && break; }
  sleep 60
done

# 2) robust latent_stats: retry with a timeout so a NCCL hang can't stall the queue forever
mkdir -p "$(dirname "$STATS")"
for attempt in 1 2 3; do
  if [ -f "$STATS" ]; then echo "### latent_stats present: $STATS"; break; fi
  port=$($PY -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
  echo "### $(date '+%F %T') latent_stats attempt $attempt (timeout 45m) port=$port"
  timeout 2700 "$TR" --nproc_per_node="$NGPU" --master-port="$port" \
    scripts/stage1/compute_latent_stats.py --config "$CFG" --data-dir /mnt/localssd/imagenet-256 \
    --output-path "$STATS" --num-samples 250000 2>&1 | tee "$ROOT/ckpt/abl-dit-b-mae-denoise-d512/latent_stats_retry${attempt}.log" || true
  pkill -9 -f "compute_latent_stats.py .*mae-denoise-k23-d512" 2>/dev/null || true
  sleep 15
done
[ -f "$STATS" ] || { echo "### FATAL: latent_stats still missing after 3 attempts -> aborting mae-d512"; exit 1; }

# 3) train (run_ablation_dits.sh skips the now-present stats step and trains 40ep)
echo "### $(date '+%F %T') latent_stats OK -> starting mae-d512 40ep training"
exec bash run_ablation_dits.sh mae-denoise-d512
