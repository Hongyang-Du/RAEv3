#!/usr/bin/env bash
# Layer-attribution suite for the p0.9 random-drop k23 decoder vs RAEv2 (official) baseline.
#   (1) value curve  (2) marginal PSNR dB+MSE  (3) Shapley   -> eval_subset_sweep_p09.py
#   (4) LOO + solo per-layer PSNR (1k images)                -> eval_layer_usage_1k.py x2
# 3 heavy jobs run in PARALLEL on GPU 0/1/2, then plots. High-precision settings.
set -uo pipefail
ROOT=/sensei-fs-3/users/hongyangd
REPO=$ROOT/RAEv3/RAEv2
PY=$ROOT/rae_env/bin/python
VAL=$ROOT/RAEv3_oldnorm/RAEv2/data_eval/imagenet-256-val.npz
P09=$ROOT/ckpt/omni-randomdrop-plain-k23-nano-p0.9/ckpt_ep016.pt
OUT=$REPO/output_p09
LOGD=$OUT/logs
mkdir -p "$OUT" "$LOGD"

cd "$REPO"
export DINOV3_REPO_DIR=$ROOT/dinov3_repo DINOV3_CKPT_DIR=$ROOT/pretrained_models/encoders/dinov3
export TORCH_HOME=$ROOT/.cache/torch HF_HOME=$ROOT/.cache/huggingface
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src"

echo "### $(date '+%F %T') START layer-attr p0.9"

# (1)(2)(3) subset-size sweep: value / marginal / Shapley  (high precision)
CUDA_VISIBLE_DEVICES=0 "$PY" src/eval_subset_sweep_p09.py \
  --num-images 512 --perms 128 --batch 64 --seed 0 \
  --val-npz "$VAL" --out "$OUT/subset_sweep.png" \
  > "$LOGD/subset_sweep.log" 2>&1 &
PID_SS=$!

# (4a) LOO+solo, RAEv2 official baseline
CUDA_VISIBLE_DEVICES=1 "$PY" src/eval_layer_usage_1k.py \
  --variant official --num-images 1000 --batch 32 --seed 0 \
  --val-npz "$VAL" --out "$OUT/layer_usage_raev2.json" \
  > "$LOGD/layer_usage_raev2.log" 2>&1 &
PID_R2=$!

# (4b) LOO+solo, p0.9 OmniRAE decoder (plain + cls surrogate: mean + fixed L23 surrogate)
CUDA_VISIBLE_DEVICES=2 "$PY" src/eval_layer_usage_1k.py \
  --variant raev2_ours --ckpt "$P09" --num-images 1000 --batch 32 --seed 0 \
  --val-npz "$VAL" --out "$OUT/layer_usage_p09.json" \
  > "$LOGD/layer_usage_p09.log" 2>&1 &
PID_P9=$!

fail=0
wait $PID_SS || { echo "### subset_sweep FAILED (see $LOGD/subset_sweep.log)"; fail=1; }
wait $PID_R2 || { echo "### layer_usage RAEv2 FAILED (see $LOGD/layer_usage_raev2.log)"; fail=1; }
wait $PID_P9 || { echo "### layer_usage p0.9 FAILED (see $LOGD/layer_usage_p09.log)"; fail=1; }

echo "### $(date '+%F %T') eval done (fail=$fail); plotting"
[ -f "$OUT/subset_sweep.json" ] && "$PY" plot_subset_marginal_p09.py 2>&1 | tail -2
{ [ -f "$OUT/layer_usage_raev2.json" ] && [ -f "$OUT/layer_usage_p09.json" ]; } \
  && "$PY" plot_layer_usage_p09.py 2>&1 | tail -2

echo "### $(date '+%F %T') ALL DONE. outputs in $OUT :"
ls -1 "$OUT"/*.png "$OUT"/*.pdf 2>/dev/null
