#!/usr/bin/env python3
"""Softgate gate-weight trajectories over training steps, parsed from
output_full/train_decoder_mls_softgate_sigreg/train.log (one `gate=[...]`
line every 50 steps). Shows the collapse to the shallowest layer (L11).

Dashed vertical line = first step with nonzero GAN loss (disc_start epoch).
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG = "output_full/train_decoder_mls_softgate_sigreg/train.log"
LAYERS = [11, 13, 15, 17, 19, 21, 23]

PAT = re.compile(
    r"ep\d+ s(\d+).*?gan=([0-9.e+-]+).*?gate=\[([0-9. ]+)\]")

steps, gates, gan_start = [], [], None
for line in open(LOG):
    m = PAT.search(line)
    if not m:
        continue
    step = int(m.group(1))
    steps.append(step)
    gates.append([float(x) for x in m.group(3).split()])
    if gan_start is None and float(m.group(2)) > 0:
        gan_start = step

steps = np.array(steps)
gates = np.array(gates)                      # [T, 7]
print(f"{len(steps)} log points, steps {steps[0]}..{steps[-1]}, GAN from s{gan_start}")

fig, ax = plt.subplots(figsize=(9, 5))
cmap = plt.get_cmap("viridis")               # shallow -> dark, deep -> light
for i, layer in enumerate(LAYERS):
    ax.plot(steps, gates[:, i], color=cmap(i / (len(LAYERS) - 1)),
            lw=1.8, label=f"L{layer}", zorder=2)

# uniform-1/7 reference drawn ON TOP of the curves
ax.axhline(1 / len(LAYERS), color="k", ls=":", lw=1.4, alpha=0.9, zorder=10)
ax.text(steps[-1], 1 / len(LAYERS), " uniform 1/7", va="bottom", ha="right",
        fontsize=8, alpha=0.9, zorder=11)

if gan_start is not None:
    ax.axvline(gan_start, color="crimson", ls="--", lw=1.5, alpha=0.8, zorder=3)
    ax.text(gan_start, ax.get_ylim()[1] * 0.97, " GAN on", color="crimson",
            va="top", fontsize=9, zorder=11)

ax.set_xlabel("training step")
ax.set_ylabel("softmax gate weight")
ax.legend(title="DINOv3 layer\n(shallow → deep)", fontsize=9, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
for ext in ("pdf", "png"):
    out = f"output_full/gate_weights_softgate.{ext}"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}")
