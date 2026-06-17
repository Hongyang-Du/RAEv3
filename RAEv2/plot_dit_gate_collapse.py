#!/usr/bin/env python3
"""Single-panel: how the stage-2 DiT's LEARNED layer gate collapses.
Parsed from ckpts_full/stage2/dit-k23-learned-gate/log.txt (gate/L* every 100 steps).
Per-layer softmax gate weight vs step, 23 curves colored DARK -> LIGHT (L1 dark,
L23 light). Free to choose, the DiT collapses onto the single deepest layer (L23).
Saves output_full/dit_gate_collapse.{png,pdf}."""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG = "ckpts_full/stage2/dit-k23-learned-gate/log.txt"
K = 23
steps, W = [], []
for line in open(LOG):
    if "gate/entropy" not in line:
        continue
    steps.append(int(re.search(r"Step (\d+)\]", line).group(1)))
    W.append([float(re.search(rf"gate/L{l}: ([0-9.]+)", line).group(1)) for l in range(1, K + 1)])
steps = np.array(steps); W = np.array(W)            # [T, 23]
print(f"{len(steps)} points, steps {steps[0]}..{steps[-1]}")

fig, ax = plt.subplots(figsize=(8, 5.2))
cmap = plt.get_cmap("viridis")                      # dark -> light : L1 dark, L23 light
for i in range(K):
    ax.plot(steps, W[:, i], color=cmap(i / (K - 1)),
            lw=3.0 if i == 22 else 1.4, alpha=0.9, zorder=5 if i == 22 else 2)
ax.axhline(1 / K, color="k", ls=":", lw=1.3, alpha=0.8, zorder=1)
ax.text(steps[-1], 1 / K, " uniform 1/23", va="bottom", ha="right", fontsize=8, alpha=0.8)
ax.text(steps[-1], W[-1, 22], f"  L23 = {W[-1, 22]:.2f}", color=cmap(1.0),
        va="center", ha="left", fontsize=11, fontweight="bold")
ax.set_xlim(steps[0], steps[-1] * 1.07)
ax.set_xlabel("DiT training step")
ax.set_ylabel("softmax gate weight")
ax.set_title("DiT learned gate collapses onto the deepest layer (L23)")
ax.grid(alpha=0.3)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, K))
cb = fig.colorbar(sm, ax=ax, pad=0.01)
cb.set_label("DINOv3 layer  (deep → shallow color: dark → light)")

fig.tight_layout()
for ext in ("pdf", "png"):
    out = f"output_full/dit_gate_collapse.{ext}"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}")
