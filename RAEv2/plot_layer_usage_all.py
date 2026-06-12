#!/usr/bin/env python3
"""Per-layer usage for the ALL-24-layer dropmean+BN stage-1 run.

Parses the LAST 'Val LOO dPSNR' / 'Val solo PSNR' lines (and the preceding
'Val PSNR (EMA)' full-mean reference) from the run's train.log and plots two
panels over L0..L23:
  left : LOO dPSNR (full - without layer i)  -> reliance on each layer
  right: solo PSNR (layer i alone)           -> sufficiency of each layer
"""
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG = sys.argv[1] if len(sys.argv) > 1 else \
    "output_full/train_decoder_mls_dropmean_bn_all24/train.log"

loo, solo, full, epoch = None, None, None, 0
for line in open(LOG):
    m = re.search(r"Val PSNR \(EMA\): ([0-9.]+) dB", line)
    if m:
        full = float(m.group(1))
        epoch += 1
    m = re.search(r"Val LOO dPSNR = \[([^\]]+)\]", line)
    if m:
        loo = [float(x) for x in m.group(1).split()]
    m = re.search(r"Val solo PSNR = \[([^\]]+)\]", line)
    if m:
        solo = [float(x) for x in m.group(1).split()]

assert loo and solo and full, f"probe lines not found in {LOG}"
layers = np.arange(len(loo))
cmap = plt.get_cmap("viridis")
colors = [cmap(i / (len(layers) - 1)) for i in layers]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.bar(layers, loo, color=colors)
ax1.axhline(0, color="k", lw=0.8)
ax1.set_xticks(layers[::2]); ax1.set_xticklabels([f"L{i}" for i in layers[::2]])
ax1.set_ylabel("LOO ΔPSNR [dB]  (full − without layer)")
ax1.set_title("Reliance: drop in PSNR when each layer is removed")
ax1.grid(alpha=0.3, axis="y")

ax2.plot(layers, solo, "o-", color="tab:purple", lw=2, ms=5)
ax2.axhline(full, color="tab:red", ls="--", lw=1.5,
            label=f"full mean (all 24): {full:.2f} dB")
for i in (0, len(layers) // 2, len(layers) - 1):
    ax2.annotate(f"{solo[i]:.1f}", (layers[i], solo[i]),
                 textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax2.set_xticks(layers[::2]); ax2.set_xticklabels([f"L{i}" for i in layers[::2]])
ax2.set_ylabel("solo PSNR [dB]  (layer alone)")
ax2.set_title("Sufficiency: reconstruction from each layer alone")
ax2.legend()
ax2.grid(alpha=0.3)

fig.suptitle(f"Per-layer decoder usage — ALL 24 DINOv3-L blocks, dropmean 0.3 + "
             f"BN projector + global SIGReg  (epoch {epoch})", fontsize=12)
fig.tight_layout()
out = "output_full/layer_usage_all24.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved -> {out}")
