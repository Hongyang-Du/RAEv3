#!/bin/bash
cd /workspace/RAEv3/RAEv2
source /opt/conda/etc/profile.d/conda.sh; conda activate rae
EVAL=data_eval/imagenet-256-val-eval.npz   # evanarlian test, center-crop (paper/table val set)

CUDA_VISIBLE_DEVICES=2 python src/eval_subset_sweep.py --num-images 1000 --batch 8 --perms 16 \
    --val-npz $EVAL --out output_full/subset_sweep.png > output_full/sweep_full.log 2>&1 &
PS=$!
CUDA_VISIBLE_DEVICES=0 python src/eval_layer_usage_1k.py --variant official --num-images 10000 --seed 0 \
    --val-npz $EVAL --out output_full/layer_usage_1k_official_train.json > output_full/lu_official.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 python src/eval_layer_usage_1k.py --variant raev2_ours --num-images 10000 --seed 0 \
    --ckpt output_full/gdrive_omnirae/omnirae_ckpt.pt \
    --val-npz $EVAL --out output_full/layer_usage_dropmean_plain_k23.json > output_full/lu_omnirae.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=3 bash -c "
  source /opt/conda/etc/profile.d/conda.sh; conda activate rae; cd /workspace/RAEv3/RAEv2
  python src/eval_feed_k7_viz.py --n-viz 40 --seed 3 --val-npz $EVAL --layers 11,13,15,17,19,21,23 --out output_full/feed_k7_pool.npz > output_full/feed_k7_regen.log 2>&1
  python src/eval_feed_k7_viz.py --n-viz 40 --seed 3 --val-npz $EVAL --layers 11 --out output_full/feed_L11_pool.npz > output_full/feed_L11_regen.log 2>&1
" &
PF=$!

wait $P1; wait $P2; python plot_layer_usage_compare_k23.py > output_full/plot_lu.log 2>&1 && echo "[PLOT] layer_usage done"
wait $PF; python plot_feed_k7_template.py --npz output_full/feed_k7_pool.npz --npz2 output_full/feed_L11_pool.npz --rows 2,9 --out output_full/feed_k7_template > output_full/plot_feed.log 2>&1 && echo "[PLOT] feed done"
wait $PS; python plot_subset_marginal.py > output_full/plot_marg.log 2>&1 && echo "[PLOT] subset_sweep_marginal done"
echo "=== REGEN ALL DONE ==="
