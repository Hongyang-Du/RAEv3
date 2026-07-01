#!/bin/bash
# 3-feed recon eval (PSNR/SSIM/rFID) for the NEW random-drop decoder
# repro-nano-randomdrop-plain-k23/ckpt_latest.pt (ep16, ema_dec), on the OFFICIAL imagenet 50k val npz.
# feed k=23 (all layers) | feed k=7 (layers 11,13,15,17,19,21,23) | feed l_11 (layer 11).
# Each feed runs on its own GPU in parallel.
set -u
ROOT=/sensei-fs-3/users/hongyangd/RAEv3/RAEv2
PY=/sensei-fs-3/users/hongyangd/rae_env/bin/python
CFG=$ROOT/configs/stage1/decoder/random-drop-layer-mls-plain-k23-nano.yaml
CKPT=/sensei-fs-3/users/hongyangd/ckpt/repro-nano-randomdrop-plain-k23/ckpt_latest.pt
REF=/sensei-fs-3/users/hongyangd/official_raev2/data/imagenet-256/imagenet-256-val.npz
OUT=/tmp/claude-1000/-sensei-fs-3-users-hongyangd/b305f2ec-30af-48de-9443-6ae01e56c9d2/scratchpad/rd3feed
mkdir -p "$OUT"
export DINOV3_REPO_DIR=/sensei-fs-3/users/hongyangd/dinov3_repo
export DINOV3_CKPT_DIR=/sensei-fs-3/users/hongyangd/pretrained_models/encoders/dinov3
export TORCH_HOME=/sensei-fs-3/users/hongyangd/.cache/torch
cd "$ROOT"

run() {  # gpu tag idxarg
  local gpu=$1 tag=$2 idx=$3
  local args=(--config "$CFG" --ckpt "$CKPT" --val-npz "$REF" --num-images 50000 \
              --batch 128 --tag "$tag" --out "$OUT/$tag.json")
  [ -n "$idx" ] && args+=(--idx "$idx")
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u "$ROOT/src/eval_recon_subset_rfid.py" "${args[@]}" \
      > "$OUT/$tag.log" 2>&1 &
  echo "GPU$gpu -> $tag (idx='${idx:-ALL}') pid $!"
}

run 0 feed_k23 ""
run 1 feed_k7  "10,12,14,16,18,20,22"
run 2 feed_l11 "10"
wait
echo "ALL 3 FEEDS DONE"
for t in feed_k23 feed_k7 feed_l11; do echo "=== $t ==="; cat "$OUT/$t.json" 2>/dev/null; echo; done
