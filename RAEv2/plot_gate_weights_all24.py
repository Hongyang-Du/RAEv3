#!/usr/bin/env python3
"""Gate-weight trajectories for the ALL-24-layer plain softgate run
(raev2 + learnable softmax gate, partial ImageNet ~93K, 5 epochs).

Parses the `gate=[...]` field (printed every 50 steps) from the run's
train.log; 24 viridis lines L0 (dark) -> L23 (yellow); dashed vertical
line = first step with nonzero GAN loss.
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "output_full/train_decoder_mls_softgate_all24"

steps, gates, gan_start = [], [], None
tsv = os.path.join(RUN_DIR, "gate_log.tsv")
if os.path.exists(tsv):
    # per-step full-precision trace
    for line in open(tsv):
        parts = line.split()
        steps.append(int(parts[0]))
        if gan_start is None and parts[1] == "1":
            gan_start = steps[-1]
        gates.append([float(x) for x in parts[2:]])
else:
    # fallback: parse the 2-decimal gate=[...] field every log_every steps
    PAT = re.compile(r"ep\d+ s(\d+).*?gan=([0-9.e+-]+).*?gate=\[([0-9. ]+)\]")
    for line in open(os.path.join(RUN_DIR, "train.log")):
        m = PAT.search(line)
        if not m:
            continue
        steps.append(int(m.group(1)))
        gates.append([float(x) for x in m.group(3).split()])
        if gan_start is None and float(m.group(2)) > 0:
            gan_start = steps[-1]

steps = np.array(steps)
gates = np.array(gates)                                   # [T, 24]
K = gates.shape[1]
print(f"{len(steps)} points, steps {steps[0]}..{steps[-1]}, K={K}, GAN from s{gan_start}")

fig, ax = plt.subplots(figsize=(11, 6))
cmap = plt.get_cmap("viridis")
for i in range(K):
    lw = 2.2 if gates[-1, i] == gates[-1].max() else 1.2
    ax.plot(steps, gates[:, i], color=cmap(i / (K - 1)), lw=lw)

# label the layers that end up dominant
order = np.argsort(gates[-1])[::-1][:4]
for i in order:
    ax.annotate(f"L{i} ({gates[-1, i]:.2f})", (steps[-1], gates[-1, i]),
                textcoords="offset points", xytext=(6, 0), fontsize=9,
                color=cmap(i / (K - 1)), va="center")

ax.axhline(1 / K, color="k", ls=":", lw=1.0, alpha=0.6)
ax.text(steps[0], 1 / K, f" uniform 1/{K}", va="bottom", fontsize=8, alpha=0.7)
if gan_start is not None:
    ax.axvline(gan_start, color="crimson", ls="--", lw=1.5, alpha=0.8)
    ax.text(gan_start, ax.get_ylim()[1] * 0.97, " GAN on", color="crimson",
            va="top", fontsize=9)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, K - 1))
cbar = fig.colorbar(sm, ax=ax, pad=0.08)
cbar.set_label("DINOv3-L layer (shallow → deep)")

ax.set_xlabel("training step")
ax.set_ylabel("softmax gate weight")
ax.set_title("Plain softgate, ALL 24 layers — does the learnable gate collapse, and to which layer?\n"
             "(raev2 recipe + gate only; no projector / SIGReg / dropout; partial ImageNet, 5 ep)",
             fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
out = "output_full/gate_weights_all24.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved -> {out}")
