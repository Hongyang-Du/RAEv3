#!/usr/bin/env python3
"""Layer-usage figure(s) from eval_layer_usage_1k.py jsons.

Usage:
    python plot_layer_usage_1k.py output_full/layer_usage_1k_official_train.json [more.json ...]
One json  -> two panels (LOO bars + solo curve) for that variant.
Multi-json -> overlaid curves for direct variant comparison.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

files = sys.argv[1:] or ["output_full/layer_usage_1k_official_train.json"]
runs = [json.load(open(f)) for f in files]
COLORS = ["tab:gray", "tab:red", "tab:blue", "tab:green"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
n_runs = len(runs)
w = 0.8 / n_runs                                # grouped-bar width
for ri, (r, c) in enumerate(zip(runs, COLORS)):
    layers = r["layers"]
    x = np.arange(len(layers))
    off = (ri - (n_runs - 1) / 2) * w
    label = f'{r["variant"]} (full {r["full"]:.2f} dB)'
    if n_runs == 1:
        cmap = plt.get_cmap("viridis")
        ax1.bar(x, r["loo_dpsnr"], 0.7, color=[cmap(i / (len(x) - 1)) for i in range(len(x))])
        ax2.bar(x, r["solo"], 0.7, color=[cmap(i / (len(x) - 1)) for i in range(len(x))])
    else:
        ax1.bar(x + off, r["loo_dpsnr"], w, color=c, alpha=0.88, label=label)
        ax2.bar(x + off, r["solo"], w, color=c, alpha=0.88, label=label)
    ax2.axhline(r["full"], color=c if n_runs > 1 else "tab:red", ls="--", lw=1.2, alpha=0.6)

L = runs[0]["layers"]
xt = np.arange(len(L))
step = 2 if len(L) > 12 else 1
ax1.axhline(0, color="k", lw=0.8)
ax1.set_xticks(xt[::step]); ax1.set_xticklabels([f"L{l}" for l in L[::step]])
ax1.set_ylabel("LOO ΔPSNR [dB]  (full − without layer)")
ax1.set_title("Reliance per layer")
ax1.grid(alpha=0.3, axis="y")
if len(runs) > 1:
    ax1.legend(fontsize=9)
ax2.set_xticks(xt[::step]); ax2.set_xticklabels([f"L{l}" for l in L[::step]])
ax2.set_ylabel("solo PSNR [dB]")
ax2.set_title("Sufficiency: each layer alone (dashed = full)")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, axis="y")

n = runs[0]["num_images"]
fig.suptitle(f"Per-layer decoder usage — {n} images, per-image PSNR "
             f"({runs[0].get('split', '?')})", fontsize=12)
fig.tight_layout()
out = "output_full/layer_usage_1k_compare.png" if len(runs) > 1 else \
    files[0].replace(".json", ".png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved -> {out}")
