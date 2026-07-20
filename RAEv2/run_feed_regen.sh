#!/bin/bash
set -e
cd /workspace/RAEv3/RAEv2
source /opt/conda/etc/profile.d/conda.sh; conda activate rae
# regenerate both feed pools with OmniRAE (ours col); n-viz 40 seed 3 -> same 40 idxs as original
CUDA_VISIBLE_DEVICES=3 python src/eval_feed_k7_viz.py --n-viz 40 --seed 3 \
    --layers 11,13,15,17,19,21,23 --out output_full/feed_k7_pool.npz > output_full/feed_k7_regen.log 2>&1
CUDA_VISIBLE_DEVICES=3 python src/eval_feed_k7_viz.py --n-viz 40 --seed 3 \
    --layers 11 --out output_full/feed_L11_pool.npz > output_full/feed_L11_regen.log 2>&1
echo "=== FEED POOLS REGEN DONE ==="
