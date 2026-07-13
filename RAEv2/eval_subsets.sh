#!/usr/bin/env bash
# Reconstruction eval (PSNR/SSIM/rFID) of a random-drop decoder on the OFFICIAL raev2
# 50k ImageNet-256 val set, for a layer subset (k23 / k7 / l11). Raw (non-EMA) weights,
# [0,1]-output decoder, FD rFID on. Parametric so it works for any drop-rate ckpt dir.
#   usage: OD=<ckpt_dir> DROP=<label> eval_subsets.sh <k23|k7|l11> <gpu>
set -uo pipefail
cd /sensei-fs-3/users/hongyangd/RAEv3/RAEv2
PY=/sensei-fs-3/users/hongyangd/rae_env/bin/python
CFG=configs/stage1/decoder/random-drop-layer-mls-plain-k23-nano-p01.yaml   # only combine.params.layers used
VAL="${VAL:-/mnt/localssd/raev2official/imagenet-256/imagenet-256-val.npz}"  # OFFICIAL 50k val
OD="${OD:?set OD to the ckpt dir}"
DROP="${DROP:?set DROP label, e.g. drop01}"
N="${N:-50000}"; LABEL="${LABEL:-50k}"
CK="$OD/ckpt_ep016.pt"

sub="$1"; gpu="$2"
case "$sub" in
  k23) IDXARG=() ;;
  k7)  IDXARG=(--idx 10,12,14,16,18,20,22) ;;
  l11) IDXARG=(--idx 10) ;;
  *) echo "usage: OD=.. DROP=.. $0 <k23|k7|l11> <gpu>"; exit 1 ;;
esac

CUDA_VISIBLE_DEVICES="$gpu" USE_EMA=0 NO_DENORM=1 FD_EVAL=1 \
  "$PY" src/eval_recon_subset_rfid.py --config "$CFG" --ckpt "$CK" --val-npz "$VAL" \
  --num-images "$N" --batch 32 --tag "${DROP}_${sub}" "${IDXARG[@]}" \
  --out "$OD/reconrfid_${DROP}_${sub}_official_${LABEL}.json"
