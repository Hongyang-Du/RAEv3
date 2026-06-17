#!/usr/bin/env python3
"""Combine the two marginal panels into ONE horizontal figure from
output_full/subset_sweep.json:
  left  : marginal d(k) in dB (PSNR domain)
  right : marginal MSE reduction  (y-axis ticks + label on the RIGHT)
Panels pulled close together. No GPU needed. Saves png + pdf."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

J = json.load(open("output_full/subset_sweep.json"))
res, LAYERS = J["results"], J["layers"]
sizes = list(range(1, len(LAYERS) + 1))
ks = sizes[1:]
colors = {"RAEv2": "gray", "RAEv2.5 (Ours)": "#F6850C"}
models = [m for m in colors if m in res]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for m in models:
    ax1.plot(ks, res[m]["marg_db"], "o-", ms=3, color=colors[m], label=m)
ax1.set_title("Marginal  d(k) = v(k) − v(k−1)  [dB]")
ax1.set_xlabel("k-th layer added"); ax1.set_ylabel("ΔPSNR [dB]")
ax1.grid(alpha=0.3); ax1.legend(fontsize=10)

for m in models:
    ax2.plot(ks, res[m]["marg_mse"], "o-", ms=3, color=colors[m], label=m)
ax2.set_title("Marginal  MSE reduction")
ax2.set_xlabel("k-th layer added"); ax2.set_ylabel("MSE(k−1) − MSE(k)")
ax2.grid(alpha=0.3); ax2.legend(fontsize=10)
ax2.yaxis.tick_right()                         # right panel: ticks + label on the RIGHT
ax2.yaxis.set_label_position("right")

fig.tight_layout()
fig.subplots_adjust(wspace=0.06)               # pull the two panels close
for ext in ("png", "pdf"):
    out = f"output_full/subset_sweep_marginal.{ext}"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved -> {out}")
