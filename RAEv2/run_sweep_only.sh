#!/bin/bash
cd /workspace/RAEv3/RAEv2
source /opt/conda/etc/profile.d/conda.sh; conda activate rae
CUDA_VISIBLE_DEVICES=2 python src/eval_subset_sweep.py --num-images 384 --batch 8 --perms 64 \
    --val-npz data_eval/imagenet-256-val-eval.npz --out output_full/subset_sweep.png > output_full/sweep_full.log 2>&1
python plot_subset_marginal.py > output_full/plot_marg.log 2>&1
echo "=== SUBSET_SWEEP DONE ==="
