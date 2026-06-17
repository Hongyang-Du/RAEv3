#!/usr/bin/env python3
"""Per-layer usage: RAEv2 K=7 vs Dropout Layers (dropmean), K=7 layers.
  left  : LOO dPSNR grouped bars (reliance)
  right : solo PSNR curves (sufficiency) + each model's full-layer PSNR as a
          horizontal line in its own color
Legend = color -> name (no PSNR numbers). Saves png + pdf."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

RUNS = [
    ("RAEv2 K=7",          "output_full/layer_usage_1k_raev2_k7.json",    "gray"),
    ("RAEv2.5 K=7 (Ours)", "output_full/layer_usage_1k_dropmean_k7.json", "#F6850C"),
]
FULL = 23.05                                   # shared full-layer PSNR (both ~equal)
data = [(name, json.load(open(f)), c) for name, f, c in RUNS]
layers = data[0][1]["layers"]
x = np.arange(len(layers))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
w = 0.4
for i, (name, d, c) in enumerate(data):
    off = (i - 0.5) * w
    ax1.bar(x + off, d["loo_dpsnr"], w, color=c)          # left: reliance bars
    ax2.plot(x, d["solo"], "o-", color=c, lw=2, ms=5, label=name)   # right: sufficiency curve

# shared full-layer PSNR: both models equal, drawn as a DASHED gray->orange
# gradient line (gray = RAEv2, orange = Ours). Manual dashes so the gradient
# stays clearly dashed.
cmap_go = LinearSegmentedColormap.from_list("gray_orange", ["gray", "#F6850C"])
x0, x1 = x[0] - 0.45, x[-1] + 0.45
dash, gap, xx = 0.18, 0.13, x[0] - 0.45
while xx < x1:
    xe = min(xx + dash, x1)
    ax2.plot([xx, xe], [FULL, FULL], color=cmap_go((xx - x0) / (x1 - x0)),
             lw=2.4, solid_capstyle="butt", zorder=5)
    xx += dash + gap
ax2.text(x[-1] + 0.4, FULL, f"RAEv2 = RAEv2.5 = {FULL:.2f} dB", color="0.25",
         va="bottom", ha="right", fontsize=10.5, fontweight="bold")

ax1.axhline(0, color="k", lw=0.8)
ax1.set_xticks(x); ax1.set_xticklabels([f"L{l}" for l in layers])
ax1.set_ylabel("LOO ΔPSNR [dB]  (full − without layer)")
ax1.set_title("Reliance per layer")
ax1.grid(alpha=0.3, axis="y")

ax2.set_xticks(x); ax2.set_xticklabels([f"L{l}" for l in layers])
ax2.set_ylabel("solo PSNR [dB]")
ax2.set_title("Sufficiency: each layer alone")
ax2.grid(alpha=0.3)
ax2.yaxis.tick_right()                         # right panel: y ticks + label on the RIGHT
ax2.yaxis.set_label_position("right")
ax2.legend(fontsize=11, loc="lower left")      # legend in the lower-left of the right panel

fig.tight_layout()
fig.subplots_adjust(wspace=0.05)               # pull the two panels close together
for ext in ("pdf", "png"):
    out = f"output_full/layer_usage_1k_compare.{ext}"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}")
