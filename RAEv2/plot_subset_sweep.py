#!/usr/bin/env python3
"""Re-plot the layer-subset value sweep as 4 SEPARATE figures from
output_full/subset_sweep.json (no GPU needed). Figure 1 (value) uses the std band
when 'std' is present in the json, else falls back to the 10/90 percentile band.
  1_value  2_marginal_db  3_marginal_mse  4_shapley"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

J = json.load(open("output_full/subset_sweep.json"))
res, LAYERS = J["results"], J["layers"]
sizes = list(range(1, len(LAYERS) + 1))
colors = {"RAEv2": "gray", "RAEv2.5 (Ours)": "#F6850C"}
models = [m for m in colors if m in res]
base = "output_full/subset_sweep"


def save(fig, suffix):
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}_{suffix}.{ext}", dpi=130, bbox_inches="tight")
    print(f"saved -> {base}_{suffix}.png")
    plt.close(fig)


# 1) value v(S) vs |S|  (band = ± std if available, else 10/90 pct)
fig, a = plt.subplots(figsize=(7.5, 5))
has_std = all("std" in res[m] for m in models)
for m in models:
    col = colors[m]; cu = np.array(res[m]["curve"])
    a.plot(sizes, cu, "o-", ms=3, color=col, label=m)
    if has_std:
        sd = np.array(res[m]["std"]); lo, hi = cu - sd, cu + sd
    else:
        lo, hi = res[m]["p10"], res[m]["p90"]
    a.fill_between(sizes, lo, hi, color=col, alpha=0.15)
band = "± std" if has_std else "10/90 pct"
a.set_title(f"Value  v(S) = PSNR  vs  |S|   (line = mean, band = {band})")
a.set_xlabel("|S| (number of layers)"); a.set_ylabel("PSNR [dB]")
a.grid(alpha=0.3); a.legend(fontsize=10)
save(fig, "1_value")

# 2) marginal d(k) in dB
fig, b = plt.subplots(figsize=(7.5, 5))
for m in models:
    b.plot(sizes[1:], res[m]["marg_db"], "o-", ms=3, color=colors[m], label=m)
b.set_title("Marginal  d(k) = v(k) - v(k-1)  [dB]   (decreasing = submodular/redundant)")
b.set_xlabel("k-th layer added"); b.set_ylabel("ΔPSNR [dB]")
b.grid(alpha=0.3); b.legend(fontsize=10)
save(fig, "2_marginal_db")

# 3) marginal MSE reduction
fig, c = plt.subplots(figsize=(7.5, 5))
for m in models:
    c.plot(sizes[1:], res[m]["marg_mse"], "o-", ms=3, color=colors[m], label=m)
c.set_title("Marginal  MSE reduction  [MSE domain]   (scale-dependence check)")
c.set_xlabel("k-th layer added"); c.set_ylabel("MSE(k-1) - MSE(k)")
c.grid(alpha=0.3); c.legend(fontsize=10)
save(fig, "3_marginal_mse")

# 4) Monte-Carlo Shapley per layer
fig, d = plt.subplots(figsize=(7.5, 5))
for m in models:
    d.plot(LAYERS, res[m]["shapley"], "o-", ms=3, color=colors[m], label=m)
d.set_title("Monte-Carlo Shapley per layer  (avg marginal contribution, dB)")
d.set_xlabel("DINOv3 layer"); d.set_ylabel("Shapley [dB]")
d.grid(alpha=0.3); d.legend(fontsize=10)
save(fig, "4_shapley")
