#!/usr/bin/env bash
# IG+CFG guidance sweep for the ep80 DiTs; gFID computed vs the ImageNet val 50k npz.
# GENERATION ONLY here (one (exp,combo) per GPU, wave-throttled across 8 GPUs).
# FID is computed afterward by fid_vs_val_batch.py (loads the 50k ref once).
#
#   N=10000 EP=80 EXPS="h1 encoder sigreg" bash run_gfid_guidance_sweep.sh
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"
PY=/sensei-fs-3/users/hongyangd/rae_env/bin/python
export DINOV3_REPO_DIR=/sensei-fs-3/users/hongyangd/dinov3_repo
export DINOV3_CKPT_DIR=/sensei-fs-3/users/hongyangd/pretrained_models/encoders/dinov3
export TORCH_HOME=/sensei-fs-3/users/hongyangd/.cache/torch

OUT=/sensei-fs-3/users/hongyangd/ckpt/gfid_guidance
TMP=$OUT/gen
mkdir -p "$TMP"
N=${N:-10000}
STEPS=${STEPS:-50}
EP=${EP:-80}
EPT=$(printf "%07d" "$EP")
IG_TMIN=${IG_TMIN:-0.10}

declare -A CFGY=(
  [h1]=configs/stage2/training/imagenet-dinov3l-h1decoder-plain-cls-k23.yaml
  [encoder]=configs/stage2/training/imagenet-dinov3l-encoder-cls-k23.yaml
  [sigreg]=configs/stage2/training/imagenet-dinov3l-sigreg-cls-k23.yaml )
declare -A DIR=(
  [h1]=/sensei-fs-3/users/hongyangd/ckpt/omnirae-dit-h1-plain-cls-k23-4node
  [encoder]=/sensei-fs-3/users/hongyangd/ckpt/omnirae-dit-encoder-cls-k23-4node
  [sigreg]=/sensei-fs-3/users/hongyangd/ckpt/omnirae-dit-sigreg-cls-k23-4node )

# combos: "ig cfg"  (3x3 grid; (1.0 1.0) = no-guidance baseline on the 50k ref)
COMBOS=( ${COMBOS_OVERRIDE:-} )
if [ ${#COMBOS[@]} -eq 0 ]; then
  COMBOS=( "1.0 1.0" "1.5 1.0" "2.0 1.0" \
           "1.0 1.5" "1.5 1.5" "2.0 1.5" \
           "1.0 2.0" "1.5 2.0" "2.0 2.0" )
fi
EXPS=( ${EXPS:-h1 encoder sigreg} )

JOBS=()
for exp in "${EXPS[@]}"; do for c in "${COMBOS[@]}"; do JOBS+=("$exp|$c"); done; done
echo "### $(date '+%F %T')  ${#JOBS[@]} gen jobs  N=$N steps=$STEPS ep$EP  ig_tmin=$IG_TMIN"

i=0
for job in "${JOBS[@]}"; do
  exp=${job%%|*}; rest=${job#*|}; ig=${rest%% *}; cfg=${rest##* }
  ckpt="${DIR[$exp]}/checkpoints/ep-${EPT}.pt"
  if [ ! -f "$ckpt" ]; then echo "MISSING $ckpt -- skip $exp ig=$ig cfg=$cfg"; continue; fi
  out="$TMP/${exp}_ig${ig}_cfg${cfg}_ep${EP}.npy"
  if [ -f "$out" ]; then echo "skip done $(basename $out)"; continue; fi
  gpu=$((i % 8))
  echo ">> GPU$gpu  $exp ig=$ig cfg=$cfg -> $(basename $out)"
  CUDA_VISIBLE_DEVICES=$gpu $PY gen_shard.py --config "${CFGY[$exp]}" --ckpt "$ckpt" \
    --num-samples "$N" --seed 42 --batch 64 --steps "$STEPS" \
    --ig-scale "$ig" --ig-tmin "$IG_TMIN" --ig-tmax 1.0 \
    --cfg-scale "$cfg" --cfg-tmin 0.0 --cfg-tmax 1.0 \
    --out "$out" > "${out%.npy}.log" 2>&1 &
  i=$((i+1))
  if [ $((i % 8)) -eq 0 ]; then echo "   ... wave wait ($i launched)"; wait; fi
done
wait
echo "### $(date '+%F %T')  ALL GEN DONE ($i jobs) -> $TMP"
