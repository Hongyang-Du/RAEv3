#!/usr/bin/env python3
"""Per-layer usage of the RAEv2 K=23 decoder (from layer_usage_1k_official_train.json):
  left  : LOO dPSNR bars, colored shallow->light, deep->dark
  right : solo PSNR per layer (curve) + full-layer PSNR (dashed, value written out)
No suptitle. Saves png + pdf."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAME = "RAEv2 K=23"
d = json.load(open("output_full/layer_usage_1k_official_train.json"))
layers = d["layers"]
loo, solo, full = d["loo_dpsnr"], d["solo"], d["full"]
x = np.arange(len(layers))

cmap = plt.get_cmap("viridis_r")                 # shallow -> light, deep -> dark
colors = [cmap(i / (len(x) - 1)) for i in x]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# left: reliance (LOO dPSNR) bars
ax1.bar(x, loo, 0.8, color=colors)
ax1.axhline(0, color="k", lw=0.8)
ax1.set_xticks(x[::2]); ax1.set_xticklabels([f"L{l}" for l in layers[::2]])
ax1.set_ylabel("LOO ΔPSNR [dB]  (full − without layer)")
ax1.set_title("Reliance per layer")
ax1.grid(alpha=0.3, axis="y")

# right: sufficiency (solo PSNR) + full-layer PSNR written out
ax2.plot(x, solo, "o-", color="0.4", lw=2, ms=5, label=f"{NAME} solo")
ax2.axhline(full, color="tab:red", ls="--", lw=1.6)
ax2.text(x[-1], full, f"{NAME} full = {full:.2f} dB", color="tab:red",
         va="bottom", ha="right", fontsize=11, fontweight="bold")
ax2.set_xticks(x[::2]); ax2.set_xticklabels([f"L{l}" for l in layers[::2]])
ax2.set_ylabel("solo PSNR [dB]")
ax2.set_title("Sufficiency: each layer alone (dashed = full)")
ax2.grid(alpha=0.3)

fig.tight_layout()
for ext in ("pdf", "png"):
    out = f"output_full/layer_usage_1k_official_train.{ext}"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}")
