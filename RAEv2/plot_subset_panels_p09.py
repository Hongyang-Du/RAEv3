#!/usr/bin/env python3
"""Re-render the 4 subset-sweep panels (value / marginal dB / marginal MSE / Shapley)
from output_p09/subset_sweep.json -- no GPU / no re-eval needed. Mirrors the inline
plotting in src/eval_subset_sweep_p09.py, kept in sync for font sizes."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

J = json.load(open("output_p09/subset_sweep.json"))
res, LAYERS = J["results"], J["layers"]
sizes = list(range(1, len(LAYERS) + 1))
colors = {"RAEv2": "gray", "p0.9 (drop0.9)": "#F6850C"}
models = [m for m in colors if m in res]
base = "output_p09/subset_sweep"


def save(fig, suffix):
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}_{suffix}.{ext}", dpi=130, bbox_inches="tight")
    print(f"saved -> {base}_{suffix}.png")
    plt.close(fig)


# 1) value v(S) vs |S|
fig, a = plt.subplots(figsize=(7.5, 5))
for m in models:
    col = colors[m]
    cu = np.array(res[m]["curve"]); sd = np.array(res[m]["std"])
    a.plot(sizes, cu, "o-", ms=3, color=col, label=m)
    a.fill_between(sizes, cu - sd, cu + sd, color=col, alpha=0.15)
a.set_title("Value  v(S) = PSNR  vs  |S|   (line = mean, band = ± std)")
a.set_xlabel("|S| (number of layers)"); a.set_ylabel("PSNR [dB]")
a.grid(alpha=0.3); a.legend()
save(fig, "1_value")

# 2) marginal d(k) in dB
fig, b = plt.subplots(figsize=(7.5, 5))
for m in models:
    b.plot(sizes[1:], res[m]["marg_db"], "o-", ms=3, color=colors[m], label=m)
b.set_title("Marginal  d(k) = v(k) - v(k-1)  [dB]   (decreasing = submodular/redundant)")
b.set_xlabel("k-th layer added"); b.set_ylabel("ΔPSNR [dB]")
b.grid(alpha=0.3); b.legend()
save(fig, "2_marginal_db")

# 3) marginal MSE reduction
fig, c = plt.subplots(figsize=(7.5, 5))
for m in models:
    c.plot(sizes[1:], res[m]["marg_mse"], "o-", ms=3, color=colors[m], label=m)
c.set_title("Marginal  MSE reduction  [MSE domain]   (scale-dependence check)")
c.set_xlabel("k-th layer added"); c.set_ylabel("MSE(k-1) - MSE(k)")
c.grid(alpha=0.3); c.legend()
save(fig, "3_marginal_mse")

# 4) Monte-Carlo Shapley per layer
fig, d = plt.subplots(figsize=(7.5, 5))
for m in models:
    d.plot(LAYERS, res[m]["shapley"], "o-", ms=3, color=colors[m], label=m)
d.set_title("Monte-Carlo Shapley per layer  (avg marginal contribution, dB)")
d.set_xlabel("DINOv3 layer"); d.set_ylabel("Shapley [dB]")
d.grid(alpha=0.3); d.legend()
save(fig, "4_shapley")
