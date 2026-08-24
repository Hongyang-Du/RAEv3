#!/usr/bin/env python3
"""Overlay LOO dPSNR + solo PSNR (per layer) for RAEv2 baseline vs p0.9 decoder.
Reads the two eval_layer_usage_1k.py jsons. No GPU needed."""
import json
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

SRC = {
    "RAEv2":          ("output_p09/layer_usage_raev2.json", "gray"),
    "p0.9 (drop0.9)": ("output_p09/layer_usage_p09.json",   "#F6850C"),
}
data = {}
for name, (path, col) in SRC.items():
    d = json.load(open(path))
    data[name] = (d["layers"], d["loo_dpsnr"], d["solo"], d["full"], col)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for name, (layers, dloo, solo, full, col) in data.items():
    ax1.plot(layers, dloo, "o-", ms=3, color=col, label=f"{name} (full={full:.2f} dB)")
    ax2.plot(layers, solo, "o-", ms=3, color=col, label=name)
ax1.set_title("Leave-one-out reliance  dPSNR = PSNR(all) - PSNR(all but i)")
ax1.set_xlabel("DINOv3 layer"); ax1.set_ylabel("LOO dPSNR [dB]")
ax1.grid(alpha=0.3); ax1.legend(); ax1.axhline(0, color="k", lw=0.6, alpha=0.4)

ax2.set_title("Solo sufficiency  PSNR(layer i alone)")
ax2.set_xlabel("DINOv3 layer"); ax2.set_ylabel("solo PSNR [dB]")
ax2.grid(alpha=0.3); ax2.legend()
ax2.yaxis.tick_right(); ax2.yaxis.set_label_position("right")

fig.tight_layout(); fig.subplots_adjust(wspace=0.08)
for ext in ("png", "pdf"):
    fig.savefig(f"output_p09/layer_usage_p09.{ext}", dpi=130, bbox_inches="tight")
print("saved -> output_p09/layer_usage_p09.png")
