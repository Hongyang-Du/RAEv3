#!/usr/bin/env python3
"""Cute, colorful 4-panel feed-k=7 comparison from output_full/feed_k7_viz.npz:
  Original | Official k7 | Official k23 (fed 7) | Ours k23 no-SIGReg (fed 7)
Each recon panel gets a bright colored frame + title, and its PSNR-vs-original
written in RED in the corner. Pick the image row with --i. Saves png + pdf."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ap = argparse.ArgumentParser()
ap.add_argument("--i", type=int, default=0, help="which image row in the npz")
ap.add_argument("--npz", default="output_full/feed_k7_viz.npz")
args = ap.parse_args()
d = np.load(args.npz)

PANELS = [
    ("original",          "Original",              "#2EC4B6"),
    ("official_k7",       "Official  k=7",         "#3A86FF"),
    ("official_k23_fed7", "Official  k=23 (fed 7)", "#C77DFF"),
    ("ours_no_sigreg",    "Ours  k=23  (no SIGReg)", "#FF9F1C"),
]

fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.6))
fig.patch.set_facecolor("#FFFDF7")
for ax, (key, title, col) in zip(axes, PANELS):
    img = np.transpose(d[f"img_{key}"][args.i], (1, 2, 0)).clip(0, 1)
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():                       # bright thick colored frame
        s.set_color(col); s.set_linewidth(6)
    ax.set_title(title, color=col, fontsize=15, fontweight="bold", pad=10)
    if key != "original":                              # red PSNR-vs-original in the corner
        p = float(d[f"psnr_{key}"][args.i])
        ax.text(0.04, 0.05, f"{p:.1f} dB", transform=ax.transAxes,
                fontsize=17, fontweight="bold", color="#E71D36", va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E71D36", lw=2, alpha=0.92))
    else:
        ax.text(0.04, 0.05, "reference", transform=ax.transAxes, fontsize=13,
                fontweight="bold", color=col, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=2, alpha=0.92))

fig.suptitle("Feed the same k=7 DINOv3 layers  →  reconstruction PSNR",
             fontsize=16, fontweight="bold", color="#3D405B", y=1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    out = f"output_full/feed_k7_cute.{ext}"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved -> {out}")
