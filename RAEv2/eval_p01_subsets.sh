#!/usr/bin/env bash
# Reconstruction eval (PSNR/SSIM/rFID) of the 16-epoch p0.1 random-drop decoder on the
# 50k ImageNet-256 val set, for the 3 layer subsets k23 / k7 / l11. Mirrors the p0.5/p0.7
# runs: raw (non-EMA) weights (USE_EMA=0), [0,1]-output decoder (NO_DENORM=1), FD rFID on.
# Runs the 3 subsets in PARALLEL on GPUs 0/1/2 (8 idle). Usage: eval_p01_subsets.sh <k23|k7|l11> <gpu>
set -uo pipefail
cd /sensei-fs-3/users/hongyangd/RAEv3/RAEv2
PY=/sensei-fs-3/users/hongyangd/rae_env/bin/python
OD=/sensei-fs-3/users/hongyangd/ckpt/omni-randomdrop-plain-k23-nano-p0.1-2node
CFG=configs/stage1/decoder/random-drop-layer-mls-plain-k23-nano-p01.yaml
VAL=/mnt/localssd/data_eval/imagenet-256-val.npz
CK="$OD/ckpt_ep016.pt"
N="${N:-50000}"
LABEL="${LABEL:-50k}"     # filename label (matches p0.5/p0.7 convention)

sub="$1"; gpu="$2"
case "$sub" in
  k23) IDXARG=() ;;                                  # all 23 layers (full mean)
  k7)  IDXARG=(--idx 10,12,14,16,18,20,22) ;;        # layers 11,13,15,17,19,21,23
  l11) IDXARG=(--idx 10) ;;                          # layer 11 only
  *) echo "usage: $0 <k23|k7|l11> <gpu>"; exit 1 ;;
esac

CUDA_VISIBLE_DEVICES="$gpu" USE_EMA=0 NO_DENORM=1 FD_EVAL=1 \
  "$PY" src/eval_recon_subset_rfid.py --config "$CFG" --ckpt "$CK" --val-npz "$VAL" \
  --num-images "$N" --batch 32 --tag "drop01_${sub}" "${IDXARG[@]}" \
  --out "$OD/reconrfid_drop01_${sub}_${LABEL}.json"
