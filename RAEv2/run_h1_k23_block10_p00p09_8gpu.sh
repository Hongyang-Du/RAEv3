#!/usr/bin/env bash
# Single-node 8xA100 sequential run of the k23 "DiT on decoder-layer h1" family
# (stage1.rae_decoder_h1.RAEDecoderH1, block_idx=10) at two decoder drop-rates,
# run SEQUENTIALLY in the order p_drop=0.0 -> p_drop=0.9.
#
# For EACH variant this does two steps (auto-skips step 1 if stats already exist):
#   1) precompute h1_stats.pt for block_idx=10 (single GPU, ~50k images)
#   2) 40-epoch DiT-B training (8 GPU, global_batch_size=2048) on that frozen h1
#
# Configs are BLOCK-AGNOSTIC (block_idx is the only knob, overridden via the
# trailing `stage_1.params.block_idx=10` dotlist arg); h1_stats_path interpolates
# to .../dit-b-h1-p0X-k23-block10/h1_stats.pt automatically.
#
# Detached run so closing the session does NOT kill training:
#   cd /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2
#   nohup bash run_h1_k23_block10_p00p09_8gpu.sh > /sensei-fs-3/users/hongyangd/logs/h1_k23_block10_p00p09.log 2>&1 &
#
# train.py auto-resumes each variant from its own <out_dir>/ckpt_latest.pt, so a
# preemption mid-run just resumes that variant (does not restart the sequence).
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"
REPO="$(pwd)"
ROOT=/sensei-fs-3/users/hongyangd

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
unset VIRTUAL_ENV

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CKPT_KEEP_RECENT="${CKPT_KEEP_RECENT:-2}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-500}"
# wandb key is rotated/invalid on this box (see run_k7_dropsweep_p09p06_8gpu.sh) -> offline
# by default so a run is never blocked on auth; sync later with `wandb sync`.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_ENTITY=uscgvl
export WANDB_PROJECT=omnirae

freeport () { ${PY} -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

DATA=/mnt/localssd/imagenet-256
echo "##### $(date '+%F %T') waiting for imagenet stage -> $DATA ..."
until [ -f "$DATA/.SYNC_DONE" ] || [ -d "$DATA/imagenet-latents-images" ]; do sleep 30; done
echo "##### $(date '+%F %T') imagenet ready ($(du -sh "$DATA" 2>/dev/null | cut -f1))"

BLOCK=10

for tag in p00 p09; do
  CFG="configs/stage2/training/imagenet-dinov3l-h1decoder-${tag}-cls-k23-ditb.yaml"
  NAME="dit-b-h1-${tag}-k23-block${BLOCK}"
  STATS="$ROOT/ckpt/${NAME}/h1_stats.pt"

  if [ ! -f "$STATS" ]; then
    echo "##### $(date '+%F %T') START stats ${tag} block${BLOCK} -> ${STATS}"
    mkdir -p "$(dirname "$STATS")"
    "$TR" --nproc_per_node=1 --master-port="$(freeport)" \
        src/compute_h1_stats.py --config "$CFG" \
        --num-samples 50000 --batch 256 --out "$STATS" \
        stage_1.params.block_idx=${BLOCK} \
        > "$ROOT/ckpt/${NAME}/h1_stats.log" 2>&1
    rc=$?
    echo "##### $(date '+%F %T') DONE stats ${tag} (exit ${rc})"
    if [ "$rc" -ne 0 ] || [ ! -f "$STATS" ]; then
      echo "##### stats precompute failed for ${tag} (exit ${rc}, stats present=$([ -f "$STATS" ] && echo yes || echo no)); stopping sequence."
      exit 1
    fi
  else
    echo "##### $(date '+%F %T') stats already present for ${tag}: ${STATS} (skip precompute)"
  fi

  echo "##### $(date '+%F %T') START train ${tag} block${BLOCK}  cfg=${CFG}  name=${NAME}  ngpu=${NGPU}"
  EXPERIMENT_NAME="${NAME}" "$TR" --nproc_per_node="${NGPU}" --master-port="$(freeport)" \
      src/train.py --config "${CFG}" \
      --results-dir "$ROOT/ckpt" --precision bf16 --wandb \
      stage_1.params.block_idx=${BLOCK}
  rc=$?
  echo "##### $(date '+%F %T') DONE train ${tag} (exit ${rc})"
  if [ "$rc" -ne 0 ]; then
    echo "##### ${tag} training failed (exit ${rc}); stopping sequence."
    exit "$rc"
  fi
done
echo "##### $(date '+%F %T') ALL DONE (p00,p09 @ block${BLOCK})"
