#!/usr/bin/env python3
"""Feed comparison, clean grid + per-column gray labels at the bottom.
7-col grouped mode (when --npz2 given):
  original | [3 decoders fed k=7] | [3 decoders fed L11]
with a box around each feeding group labelled "fed k=7 layers" / "fed L11 only".
Without --npz2: original | 3 decoders fed k=7. Per-cell PSNR-vs-original in red."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="2,9", help="comma-separated image indices")
ap.add_argument("--npz", default="output_full/feed_k7_pool.npz")
ap.add_argument("--npz2", default=None, help="2nd npz (L11): adds the 2nd feeding group")
ap.add_argument("--out", default="output_full/feed_k7_template")
args = ap.parse_args()
d = np.load(args.npz)
d2 = np.load(args.npz2) if args.npz2 else None
rows = [int(x) for x in args.rows.split(",")]

DEC = [("official_k7", "RAEv2 K=7"), ("official_k23_fed7", "RAEv2 K=23"),
       ("ours_no_sigreg", "RAEv2.5 K=23")]
# (source, key, bottom-label, group)
COLS = [(d, "original", "original", None)]
COLS += [(d, k, l, "k7") for k, l in DEC]
if d2 is not None:
    COLS += [(d2, k, l, "l11") for k, l in DEC]

nr, nc = len(rows), len(COLS)
fig, axes = plt.subplots(nr, nc, figsize=(2.15 * nc, 2.15 * nr))
axes = np.atleast_2d(axes)
for r, i in enumerate(rows):
    for c, (src, key, lab, _) in enumerate(COLS):
        ax = axes[r, c]
        ax.imshow(np.transpose(src[f"img_{key}"][i], (1, 2, 0)).clip(0, 1))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if key != "original":
            p = float(src[f"psnr_{key}"][i])
            ax.text(0.05, 0.95, f"{p:.1f}", transform=ax.transAxes, fontsize=11,
                    fontweight="bold", color="#E71D36", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8))
        if r == nr - 1:
            ax.set_xlabel(lab, fontsize=10.5, color="0.35", labelpad=6)

fig.subplots_adjust(wspace=0.03, hspace=0.03, top=0.9, bottom=0.07, left=0.01, right=0.99)

# group boxes + top labels
def group_box(cols, label, color):
    p0 = axes[0, cols[0]].get_position(); p1 = axes[0, cols[-1]].get_position()
    pb = axes[-1, cols[0]].get_position()
    x0, x1, y0, y1 = p0.x0, p1.x1, pb.y0, p0.y1
    pad = 0.006
    fig.add_artist(Rectangle((x0 - pad, y0 - pad), (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad,
                             transform=fig.transFigure, fill=False, ec=color, lw=2.4, zorder=20))
    fig.text((x0 + x1) / 2, y1 + pad + 0.012, label, ha="center", va="bottom",
             fontsize=14, fontweight="bold", color=color)

if d2 is not None:
    group_box([1, 2, 3], "fed k=7 layers", "#3A86FF")
    group_box([4, 5, 6], "fed L11 only", "#FF9F1C")

for ext in ("png", "pdf"):
    fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", dpi=150)
    print(f"saved -> {args.out}.{ext}")
