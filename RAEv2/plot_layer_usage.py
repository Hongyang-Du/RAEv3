#!/usr/bin/env python3
"""Layer-usage comparison figure for the three stage-1 decoders (all at epoch 5).

Data provenance (5 fixed val images, assets/samples, Resize288+CenterCrop256):
  - raev2:    probe with its own combine (subset mean + fixed L23 token-mean surrogate),
              CPU run 2026-06-11, ema_dec from output_full/train_decoder_mls_raev2
  - nogate:   src/probe_layer_usage.py on output_full/train_decoder_mls_nogate_sigreg
  - dropmean: final-epoch Val LOO/solo lines in
              output_full/train_decoder_mls_dropmean_sigreg/train.log

LOO dPSNR_i = full - PSNR(without layer i)  -> reliance on layer i (gate-weight analog)
solo PSNR_i = PSNR(layer i alone)           -> sufficiency of layer i
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAYERS = [11, 13, 15, 17, 19, 21, 23]

RUNS = {
    "raev2 (MLS)": dict(
        full=25.98,
        loo=[+2.81, +0.89, +0.12, +0.12, +0.20, +0.36, +0.74],
        solo=[20.21, 21.54, 23.81, 20.15, 14.38, 12.12, 10.82],
        color="tab:gray",
    ),
    "nogate + SIGReg": dict(
        full=26.02,
        loo=[+3.80, +0.89, +0.25, +0.33, +0.34, +0.57, +0.19],
        solo=[20.90, 21.92, 23.31, 16.13, 13.32, 11.11, 10.00],
        color="tab:blue",
    ),
    "dropmean + SIGReg": dict(
        full=25.80,
        loo=[+0.99, +0.43, +0.16, -0.14, -0.25, -0.29, -0.14],
        solo=[26.92, 25.91, 25.02, 22.88, 21.15, 19.72, 17.45],
        color="tab:red",
    ),
}

x = np.arange(len(LAYERS))
w = 0.26
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for i, (name, d) in enumerate(RUNS.items()):
    off = (i - 1) * w
    ax1.bar(x + off, d["loo"], w, label=name, color=d["color"], alpha=0.85)
    ax2.bar(x + off, d["solo"], w, label=f'{name} (full {d["full"]:.2f})',
            color=d["color"], alpha=0.85)
    ax2.axhline(d["full"], color=d["color"], ls="--", lw=1.0, alpha=0.6)

ax1.axhline(0, color="k", lw=0.8)
ax1.set_xticks(x); ax1.set_xticklabels([f"L{l}" for l in LAYERS])
ax1.set_ylabel("LOO ΔPSNR [dB]   (reliance on layer)")
ax1.set_title("Leave-one-out: how much the decoder relies on each layer")
ax1.legend()
ax1.grid(alpha=0.3, axis="y")

ax2.set_xticks(x); ax2.set_xticklabels([f"L{l}" for l in LAYERS])
ax2.set_ylabel("solo PSNR [dB]   (layer alone)")
ax2.set_title("Solo decoding: each layer's sufficiency (dashed = full-mean PSNR)")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, axis="y")

fig.suptitle("Per-layer decoder usage, DINOv3-L K7 — stage-1 @ epoch 5 "
             "(learnable gates collapse; dropmean flattens reliance & makes every layer decodable)",
             fontsize=11)
fig.tight_layout()
out = "output_full/layer_usage_compare.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved -> {out}")
