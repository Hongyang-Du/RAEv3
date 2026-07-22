#!/usr/bin/env bash
# k23/k7/l11 recon eval (PSNR+SSIM) for the 3 5-epoch sweep ckpts (anchor, Variant A
# maskcond, Variant B depthattn). Mirrors RAEv3_oldnorm/README.md's "Eval protocol":
# the conditioning/fusion mask must equal the mask that built z -- eval_recon_subset.py
# auto-detects mask-cond ckpts and passes the matched k-hot mask for --idx.
#
# Per-feed idx (0-based positions into the trained layers=[1..23] list):
#   k23full = full feed (no --idx)
#   k7      = idx 0,1,2,3,4,5,6   (shallowest-7 prefix feed)
#   l11     = idx 10              (layer 11 solo)
#
# Standalone-runnable: bash run_variant_sweep_5ep_eval.sh
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"
ROOT=/sensei-fs-3/users/hongyangd
PY="$ROOT/rae_env/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"

NAMES=(anchor maskcond depthattn)
declare -A CFG=(
  [anchor]="configs/stage1/decoder/randomdrop-plain-k23-nano-p03-5ep-oldnorm.yaml"
  [maskcond]="configs/stage1/decoder/randomdrop-plain-k23-nano-p03-maskcond-5ep-oldnorm.yaml"
  [depthattn]="configs/stage1/decoder/randomdrop-plain-k23-nano-p03-depthattn-5ep-oldnorm.yaml"
)
declare -A OUTDIR=(
  [anchor]="$ROOT/ckpt/omni-randomdrop-plain-k23-nano-p0.3-5ep-sweep"
  [maskcond]="$ROOT/ckpt/omni-randomdrop-plain-k23-nano-p0.3-maskcond-5ep-sweep"
  [depthattn]="$ROOT/ckpt/omni-randomdrop-plain-k23-nano-p0.3-depthattn-5ep-sweep"
)

for NAME in "${NAMES[@]}"; do
  CKPT="${OUTDIR[$NAME]}/ckpt_latest.pt"
  if [ ! -f "$CKPT" ]; then
    echo "SKIP ${NAME}: no ckpt at $CKPT"
    continue
  fi
  CFGFILE="${CFG[$NAME]}"
  echo "##### $(date '+%F %T')  EVAL ${NAME}  ckpt=${CKPT}"

  "$PY" src/eval_recon_subset.py --config "$CFGFILE" --ckpt "$CKPT" \
      --tag "${NAME}_k23full"
  "$PY" src/eval_recon_subset.py --config "$CFGFILE" --ckpt "$CKPT" \
      --idx 0,1,2,3,4,5,6 --tag "${NAME}_k7"
  "$PY" src/eval_recon_subset.py --config "$CFGFILE" --ckpt "$CKPT" \
      --idx 10 --tag "${NAME}_l11"

  if [ "$NAME" = "maskcond" ]; then
    echo "##### $(date '+%F %T')  EVAL ${NAME} null-cond (net conditioning contribution)"
    "$PY" src/eval_recon_subset.py --config "$CFGFILE" --ckpt "$CKPT" \
        --null-cond --tag "${NAME}_k23full_null"
  fi
done

echo "##### $(date '+%F %T')  EVAL SWEEP DONE"
