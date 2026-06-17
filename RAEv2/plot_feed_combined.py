#!/usr/bin/env python3
"""Combine feed-k=7 and feed-L11 into ONE figure for a single original image.
Shared original (spans both rows). Rows = feeding regime (feed k=7 / feed L11);
columns = RAEv2 K=7 | RAEv2 K=23 | RAEv2.5 K=23. RAEv2 K=7 in the feed-k=7 row is
its native decoder -> tagged "naive". Per-cell PSNR-vs-original in red. png + pdf."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--i", type=int, default=26, help="image index in the pools")
ap.add_argument("--out", default="output_full/feed_combined")
args = ap.parse_args()

P = {
    "feed k=7":  np.load("output_full/feed_k7_pool.npz"),
    "feed L=11": np.load("output_full/feed_L11_pool.npz"),
}
ROWS = ["feed k=7", "feed L=11"]
COLS = [("official_k7", "RAEv2 K=7"), ("official_k23_fed7", "RAEv2 K=23"),
        ("ours_no_sigreg", "RAEv2.5 K=23")]
i = args.i

fig = plt.figure(figsize=(11.6, 6.0))
gs = fig.add_gridspec(2, 4, wspace=0.03, hspace=0.03)
ax_o = fig.add_subplot(gs[:, 0])
ax_o.imshow(np.transpose(P[ROWS[0]][f"img_original"][i], (1, 2, 0)).clip(0, 1))
ax_o.set_xticks([]); ax_o.set_yticks([])
ax_o.set_title("original", fontsize=14, color="#2EC4B6", fontweight="bold", pad=8)

for r, reg in enumerate(ROWS):
    d = P[reg]
    for c, (key, name) in enumerate(COLS):
        ax = fig.add_subplot(gs[r, c + 1])
        ax.imshow(np.transpose(d[f"img_{key}"][i], (1, 2, 0)).clip(0, 1))
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:                                   # column headers on the top row
            ax.set_title(name, fontsize=14, fontweight="bold", color="#3D405B", pad=8)
        if c == 0:                                   # row = feeding regime, left label
            ax.set_ylabel(reg, fontsize=13, fontweight="bold", color="#3D405B")
        p = float(d[f"psnr_{key}"][i])
        ax.text(0.05, 0.95, f"{p:.1f} dB", transform=ax.transAxes, fontsize=13,
                fontweight="bold", color="#E71D36", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        if reg == "feed k=7" and key == "official_k7":   # native decoder here
            ax.text(0.95, 0.05, "naive", transform=ax.transAxes, fontsize=12,
                    fontweight="bold", color="white", va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#3D405B", ec="none", alpha=0.9))

for ext in ("png", "pdf"):
    fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", dpi=150)
    print(f"saved -> {args.out}.{ext}")
