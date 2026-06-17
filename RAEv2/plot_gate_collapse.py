#!/usr/bin/env python3
"""How the stage-2 DiT's LEARNED layer gate collapses, parsed from
ckpts_full/stage2/dit-k23-learned-gate/log.txt (gate/L* + gate/entropy + gate/max
logged every 100 steps). Free to pick its own layers, the DiT throws away 22 of 23
DINOv3 layers and generates from the single deepest layer (L23).
  left  : per-layer softmax gate weight vs step (23 curves, shallow->deep), L23 wins
  right : entropy (log K -> 0) and max weight (-> 1) = collapse to one-hot
Saves png + pdf."""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG = "ckpts_full/stage2/dit-k23-learned-gate/log.txt"
K = 23
steps, W, ent, mx = [], [], [], []
for line in open(LOG):
    m = re.search(r"Step (\d+)\].*gate/entropy", line)
    if not m:
        continue
    steps.append(int(m.group(1)))
    W.append([float(re.search(rf"gate/L{l}: ([0-9.]+)", line).group(1)) for l in range(1, K + 1)])
    ent.append(float(re.search(r"gate/entropy: ([0-9.]+)", line).group(1)))
    mx.append(float(re.search(r"gate/max: ([0-9.]+)", line).group(1)))
steps = np.array(steps); W = np.array(W)           # [T, 23]
print(f"{len(steps)} points, steps {steps[0]}..{steps[-1]}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
cmap = plt.get_cmap("viridis")                      # shallow -> dark, deep -> light
for i in range(K):
    lyr = i + 1
    if lyr == 23:
        continue
    ax1.plot(steps, W[:, i], color=cmap(i / (K - 1)), lw=1.4, alpha=0.85, zorder=2)
# highlight the winner L23
ax1.plot(steps, W[:, 22], color="#F6850C", lw=3.0, zorder=5, label="L23 (winner)")
ax1.axhline(1 / K, color="k", ls=":", lw=1.3, alpha=0.8, zorder=1)
ax1.text(steps[-1], 1 / K, " uniform 1/23", va="bottom", ha="right", fontsize=8, alpha=0.8)
ax1.text(steps[-1], W[-1, 22], f" L23 = {W[-1, 22]:.2f}", color="#F6850C",
         va="center", ha="right", fontsize=10, fontweight="bold")
ax1.set_xlabel("DiT training step"); ax1.set_ylabel("softmax gate weight")
ax1.set_title("Learned gate weight per DINOv3 layer (other 22 → 0)")
ax1.legend(loc="center right", fontsize=10); ax1.grid(alpha=0.3)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, K))
cb = fig.colorbar(sm, ax=ax1, pad=0.01); cb.set_label("DINOv3 layer (shallow → deep)")

ax2.plot(steps, ent, "o-", color="#36ADA3", lw=2, ms=4, label="entropy  H(gate)")
ax2.axhline(np.log(K), color="#36ADA3", ls="--", lw=1.2, alpha=0.7)
ax2.text(steps[-1], np.log(K), f" uniform = ln 23 = {np.log(K):.2f}", color="#36ADA3",
         va="bottom", ha="right", fontsize=8)
ax2.set_xlabel("DiT training step"); ax2.set_ylabel("entropy  H(gate)", color="#36ADA3")
ax2.tick_params(axis="y", labelcolor="#36ADA3")
ax2b = ax2.twinx()
ax2b.plot(steps, mx, "s-", color="#9C27B0", lw=2, ms=4, label="max weight")
ax2b.set_ylabel("max gate weight", color="#9C27B0"); ax2b.tick_params(axis="y", labelcolor="#9C27B0")
ax2b.set_ylim(0, 1.02)
ax2.set_title("Collapse to one-hot: entropy → 0, max weight → 1")
ax2.grid(alpha=0.3)

fig.tight_layout()
for ext in ("pdf", "png"):
    out = f"output_full/gate_collapse.{ext}"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}")
