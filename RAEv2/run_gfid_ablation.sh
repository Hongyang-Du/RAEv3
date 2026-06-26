#!/usr/bin/env bash
# gFID + IS convergence ablation: for each DiT experiment, eval every-10-epoch ckpt.
# Decoder is each experiment's own stage_1 (h1/encoder -> plain k23; sigreg -> sigreg k23).
# Runs <=8 jobs at a time (one per GPU). Results -> ckpt/gfid_ablation/<exp>_ep<E>.json
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"
PY=/sensei-fs-3/users/hongyangd/rae_env/bin/python
CKPT=/sensei-fs-3/users/hongyangd/ckpt
OUT=$CKPT/gfid_ablation
mkdir -p "$OUT"
N=${N:-10000}        # samples for FID/IS (override: N=5000 bash run_gfid_ablation.sh)

# exp -> config + ckpt dir
declare -A CFG=(
  [h1]=configs/stage2/training/imagenet-dinov3l-h1decoder-plain-cls-k23.yaml
  [encoder]=configs/stage2/training/imagenet-dinov3l-encoder-cls-k23.yaml
  [sigreg]=configs/stage2/training/imagenet-dinov3l-sigreg-cls-k23.yaml
)
declare -A DIR=(
  [h1]=$CKPT/omnirae-dit-h1-plain-cls-k23-4node
  [encoder]=$CKPT/omnirae-dit-encoder-cls-k23-4node
  [sigreg]=$CKPT/omnirae-dit-sigreg-cls-k23-4node
)

# build job list: "exp epoch" for every available every-10 ckpt
JOBS=()
for exp in h1 encoder sigreg; do
  for ep in $(ls "${DIR[$exp]}/checkpoints/"ep-*.pt 2>/dev/null | grep -oE "ep-[0-9]+" | sort -u); do
    e=$((10#${ep#ep-}))
    if [ $e -gt 0 ] && [ $((e % 10)) -eq 0 ]; then JOBS+=("$exp $e"); fi
  done
done
echo "### $(date '+%F %T')  ${#JOBS[@]} eval jobs, N=$N samples, 8 GPUs"

i=0
for job in "${JOBS[@]}"; do
  set -- $job; exp=$1; e=$2
  gpu=$((i % 8))
  ckpt=$(printf "%s/checkpoints/ep-%07d.pt" "${DIR[$exp]}" "$e")
  outjson="$OUT/${exp}_ep$(printf %04d $e).json"
  if [ -f "$outjson" ]; then echo "skip $exp ep$e (done)"; i=$((i+1)); continue; fi
  echo ">> GPU$gpu  $exp ep$e -> $outjson"
  CUDA_VISIBLE_DEVICES=$gpu DINOV3_REPO_DIR=/sensei-fs-3/users/hongyangd/dinov3_repo \
    DINOV3_CKPT_DIR=/sensei-fs-3/users/hongyangd/pretrained_models/encoders/dinov3 \
    TORCH_HOME=/sensei-fs-3/users/hongyangd/.cache/torch \
    $PY src/eval_fid_dit.py --config "${CFG[$exp]}" --ckpt "$ckpt" \
      --data ./data/imagenet-256 --raw --num-samples $N --batch 64 --steps 50 \
      --out "$outjson" > "$OUT/${exp}_ep$(printf %04d $e).log" 2>&1 &
  i=$((i+1))
  # throttle: after every 8 launches, wait for the wave to finish
  if [ $((i % 8)) -eq 0 ]; then echo "   ... waiting for wave"; wait; fi
done
wait
echo "### $(date '+%F %T')  ALL DONE -> $OUT"
