#!/usr/bin/env bash
# Fast gFID+IS for a set of (exp,epoch): shard the N-sample generation across ALL 8 GPUs,
# then one combine per exp (shared real reference, cached). Usage:
#   bash run_gfid_fast.sh <epoch>        # e.g. 50  -> evals h1/encoder/sigreg at that epoch
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"
PY=/sensei-fs-3/users/hongyangd/rae_env/bin/python
CKPT=/sensei-fs-3/users/hongyangd/ckpt; OUT=$CKPT/gfid_ablation; TMP=$OUT/shards
mkdir -p "$TMP"
export DINOV3_REPO_DIR=/sensei-fs-3/users/hongyangd/dinov3_repo
export DINOV3_CKPT_DIR=/sensei-fs-3/users/hongyangd/pretrained_models/encoders/dinov3
export TORCH_HOME=/sensei-fs-3/users/hongyangd/.cache/torch
N=${N:-10000}
EP=${1:?usage: bash run_gfid_fast.sh <epoch>}
EPT=$(printf "%07d" "$EP"); EP4=$(printf "%04d" "$EP")

declare -A CFG=(
  [h1]=configs/stage2/training/imagenet-dinov3l-h1decoder-plain-cls-k23.yaml
  [encoder]=configs/stage2/training/imagenet-dinov3l-encoder-cls-k23.yaml
  [sigreg]=configs/stage2/training/imagenet-dinov3l-sigreg-cls-k23.yaml)
declare -A DIR=(
  [h1]=$CKPT/omnirae-dit-h1-plain-cls-k23-4node
  [encoder]=$CKPT/omnirae-dit-encoder-cls-k23-4node
  [sigreg]=$CKPT/omnirae-dit-sigreg-cls-k23-4node)
# GPU allocation across the 3 exps (8 GPUs): h1=0,1,2  encoder=3,4,5  sigreg=6,7
declare -A GPUS=( [h1]="0 1 2" [encoder]="3 4 5" [sigreg]="6 7" )

echo "### $(date '+%F %T')  fast gFID ep$EP  N=$N  (shards across 8 GPUs)"
declare -A SHARDS
for exp in h1 encoder sigreg; do
  ckpt="${DIR[$exp]}/checkpoints/ep-${EPT}.pt"
  [ -f "$ckpt" ] || { echo "MISSING $ckpt — skip $exp"; continue; }
  gpus=(${GPUS[$exp]}); ng=${#gpus[@]}
  ss=$(( (N + ng - 1) / ng ))           # ceil(N/ng) per shard
  paths=""
  for j in "${!gpus[@]}"; do
    g=${gpus[$j]}; out="$TMP/${exp}_ep${EP4}_s${j}.npy"; paths="${paths:+$paths,}$out"
    CUDA_VISIBLE_DEVICES=$g $PY gen_shard.py --config "${CFG[$exp]}" --ckpt "$ckpt" \
      --num-samples "$ss" --seed $((1000 + 17*j)) --batch 64 --steps 50 --out "$out" \
      > "$TMP/${exp}_ep${EP4}_s${j}.log" 2>&1 &
  done
  SHARDS[$exp]="$paths"
done
echo "waiting for all generation shards..."; wait
echo "### $(date '+%F %T')  generation done; computing FID+IS"

# combines (sequential; first builds the shared reference cache)
for exp in h1 encoder sigreg; do
  [ -n "${SHARDS[$exp]:-}" ] || continue
  CUDA_VISIBLE_DEVICES=0 $PY fid_is_combine.py --config "${CFG[$exp]}" \
    --shards "${SHARDS[$exp]}" --data ./data/imagenet-256 --total "$N" \
    --ref-num "$N" --ref-seed 42 --ref-npy "$TMP/ref_${N}.npy" --epoch "$EP" \
    --out "$OUT/${exp}_ep${EP4}.json" 2>&1 | grep -ivE "QWEN3p5|reference [0-9]"
done
echo "### $(date '+%F %T')  DONE ep$EP"
for exp in h1 encoder sigreg; do
  [ -f "$OUT/${exp}_ep${EP4}.json" ] && echo "  $exp ep$EP: $(cat $OUT/${exp}_ep${EP4}.json | tr -d '\n ' )"
done
