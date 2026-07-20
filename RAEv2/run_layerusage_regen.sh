#!/bin/bash
set -e
cd /workspace/RAEv3/RAEv2
source /opt/conda/etc/profile.d/conda.sh; conda activate rae
# RAEv2 baseline = OFFICIAL k23 (unchanged loader); OmniRAE = GDrive ckpt via raev2_ours (mean + L23 surrogate, in-distribution for cls-on)
CUDA_VISIBLE_DEVICES=0 python src/eval_layer_usage_1k.py --variant official \
    --out output_full/layer_usage_1k_official_train.json > output_full/lu_official.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 python src/eval_layer_usage_1k.py --variant raev2_ours \
    --ckpt output_full/gdrive_omnirae/omnirae_ckpt.pt \
    --out output_full/layer_usage_dropmean_plain_k23.json > output_full/lu_omnirae.log 2>&1 &
P2=$!
wait $P1; wait $P2
echo "=== LAYER USAGE EVALS DONE ==="
