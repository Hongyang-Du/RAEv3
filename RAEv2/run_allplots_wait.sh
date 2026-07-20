#!/bin/bash
cd /workspace/RAEv3/RAEv2
source /opt/conda/etc/profile.d/conda.sh; conda activate rae

# 1) layer-usage compare (fast) -> wait for both jsons, then plot
until grep -q "LAYER USAGE EVALS DONE" output_full/lu_driver.log 2>/dev/null; do sleep 10; done
python plot_layer_usage_compare_k23.py > output_full/plot_lu.log 2>&1 && echo "[PLOT] layer_usage_compare_k23 done"

# 2) feed template (fast) -> wait for both pools, then plot with both feeding groups
until grep -q "FEED POOLS REGEN DONE" output_full/feed_driver.log 2>/dev/null; do sleep 10; done
python plot_feed_k7_template.py --npz output_full/feed_k7_pool.npz \
    --npz2 output_full/feed_L11_pool.npz --rows 2,9 --out output_full/feed_k7_template \
    > output_full/plot_feed.log 2>&1 && echo "[PLOT] feed_k7_template done"

# 3) subset sweep (slow) -> 4 figs come from the eval itself; wait then plot the marginal combo
until [ -f output_full/subset_sweep.json ] && grep -q "subset_sweep_4_shapley.png" output_full/sweep_full.log 2>/dev/null; do sleep 15; done
python plot_subset_marginal.py > output_full/plot_marg.log 2>&1 && echo "[PLOT] subset_sweep_marginal done"

echo "=== ALL PLOTS DONE ==="
