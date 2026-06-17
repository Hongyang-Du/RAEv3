#!/usr/bin/env python3
"""Reconstruction PSNR over training steps: softgate (learnable softmax gate over
the raev2 K=7 layers, collapses to L11) vs raev2 K=7 (fixed uniform mean over the
same 7 layers). Per-step train-batch PSNR parsed from each train.log."""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = {
    "softgate (learnable gate, K=7)": ("output_full/train_decoder_mls_softgate_sigreg/train.log", "tab:red"),
    "raev2 K=7 (uniform mean)":       ("output_full/train_decoder_mls_raev2/train.log",            "tab:gray"),
}
PAT = re.compile(r"ep\d+ s(\d+).*?psnr=([0-9.]+)")
GANPAT = re.compile(r"ep\d+ s(\d+).*?gan=([0-9.e+-]+)")


def parse(path):
    steps, psnr, gan_start = [], [], None
    for line in open(path):
        m = PAT.search(line)
        if m:
            steps.append(int(m.group(1))); psnr.append(float(m.group(2)))
        g = GANPAT.search(line)
        if g and gan_start is None and float(g.group(2)) > 0:
            gan_start = int(g.group(1))
    return np.array(steps), np.array(psnr), gan_start


fig, ax = plt.subplots(figsize=(9, 5))
gan_marks = []
for name, (path, color) in RUNS.items():
    s, p, gs = parse(path)
    ax.plot(s, p, color=color, lw=1.6, label=f"{name}  (last {p[-1]:.2f} dB)")
    if gs is not None:
        gan_marks.append(gs)
    print(f"{name}: {len(s)} pts, steps {s[0]}..{s[-1]}, last PSNR {p[-1]:.2f}")

if gan_marks:
    gs = min(gan_marks)
    ax.axvline(gs, color="k", ls="--", lw=1.0, alpha=0.5, zorder=1)
    ax.text(gs, ax.get_ylim()[0] + 0.3, " GAN on", fontsize=8, alpha=0.6)

ax.set_xlabel("training step")
ax.set_ylabel("reconstruction PSNR [dB]")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
for ext in ("pdf", "png"):
    out = f"output_full/psnr_softgate_vs_k7.{ext}"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}")
