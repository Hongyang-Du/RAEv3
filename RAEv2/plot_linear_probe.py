#!/usr/bin/env python3
"""Linear-probe top-1 comparison figure, from output_full/linear_probes/results.json
(train_linear_probes.py). Curve: each DINOv3-L layer alone, shallow->deep.
Dashed horizontal lines: the combined latents the decoders/DiTs actually consume."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

res = json.load(open("output_full/linear_probes/results.json"))
acc = res["history"][-1]                      # final epoch
layers = res["layers"]
epoch = acc["epoch"]

# All combined latents WITHOUT the CLS surrogate (L23 token-mean), for an
# apples-to-apples comparison of the spatial token spaces. Note: raev2's MLS
# minus the surrogate IS the plain 7-layer mean, so it reuses the mls_mean head.
COMBOS = [  # key, label, color
    ("mls_mean", "raev2 MLS mean, no CLS (= proj input)", "tab:gray"),
    ("nogate",   "nogate + SIGReg (proj out)",            "tab:blue"),
    ("dropmean", "dropmean + SIGReg (proj out)",          "tab:red"),
]

fig, ax = plt.subplots(figsize=(9, 5.5))

vals = [acc[f"L{l}"] for l in layers]
ax.plot(layers, vals, "o-", color="tab:purple", lw=2, ms=6,
        label="single DINOv3 layer (shallow → deep)")
for l, v in zip(layers, vals):
    ax.annotate(f"{v:.1f}", (l, v), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=9)

for key, label, color in COMBOS:
    v = acc[key]
    ax.axhline(v, color=color, ls="--", lw=1.6, alpha=0.85,
               label=f"{label}: {v:.1f}")

ax.set_xticks(layers)
ax.set_xticklabels([f"L{l}" for l in layers])
ax.set_xlabel("DINOv3-L layer")
ax.set_ylabel("ImageNet val top-1 [%]  (linear probe, 25k held-out)")
ax.set_title("Semantic content: linear probe on pooled tokens\n"
             f"per-layer curve vs combined latents (dashed)  —  epoch {epoch}",
             fontsize=11)
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
out = "output_full/linear_probe_compare.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved -> {out}")
