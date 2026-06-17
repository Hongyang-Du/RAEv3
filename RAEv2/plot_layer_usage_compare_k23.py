#!/usr/bin/env python3
"""Per-layer usage, K=23: RAEv2 (gray) vs our Random-Drop / NO-SIGReg = RAEv2.5 (orange).
  left  : LOO dPSNR grouped bars (reliance) — both models, for comparison
  right : solo PSNR curves (sufficiency) + each model's full-layer PSNR as a
          horizontal dashed line in its own color, value written out (dB).
Both fulls are read from the JSONs (NOT hardcoded). Saves png + pdf."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = [
    ("RAEv2",        "output_full/layer_usage_1k_official_train.json",   "gray"),
    ("RAEv2.5 (Ours)", "output_full/layer_usage_dropmean_plain_k23.json", "#F6850C"),
]
data = [(name, json.load(open(f)), c) for name, f, c in RUNS]
layers = data[0][1]["layers"]
x = np.arange(len(layers))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
w = 0.4
for i, (name, d, c) in enumerate(data):
    off = (i - 0.5) * w
    ax1.bar(x + off, d["loo_dpsnr"], w, color=c, label=name)            # left: reliance bars
    ax2.plot(x, d["solo"], "o-", color=c, lw=2, ms=4, label=name)       # right: sufficiency curve

# each model's full-layer PSNR: dashed line in its own color + value written out
for name, d, c in data:
    full = d["full"]
    ax2.axhline(full, color=c, ls="--", lw=1.8, zorder=5)
    ax2.text(x[-1], full, f"{name} full = {full:.2f} dB", color=c,
             va="bottom", ha="right", fontsize=10.5, fontweight="bold")

ax1.axhline(0, color="k", lw=0.8)
ax1.set_xticks(x[::2]); ax1.set_xticklabels([f"L{l}" for l in layers[::2]])
ax1.set_ylabel("LOO ΔPSNR [dB]  (full − without layer)")
ax1.set_title("Reliance per layer")
ax1.legend(fontsize=11, loc="upper right")
ax1.grid(alpha=0.3, axis="y")

ax2.set_xticks(x[::2]); ax2.set_xticklabels([f"L{l}" for l in layers[::2]])
ax2.set_ylabel("solo PSNR [dB]")
ax2.set_title("Sufficiency: each layer alone (dashed = full)")
ax2.grid(alpha=0.3)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.legend(fontsize=11, loc="lower left")

fig.tight_layout()
fig.subplots_adjust(wspace=0.05)
for ext in ("pdf", "png"):
    out = f"output_full/layer_usage_compare_k23.{ext}"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}")
