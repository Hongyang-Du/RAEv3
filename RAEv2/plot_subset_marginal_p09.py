#!/usr/bin/env python3
"""marginal panels (dB + MSE) for the p0.9 subset sweep. Reads output_p09/subset_sweep.json."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

J = json.load(open("output_p09/subset_sweep.json"))
res, LAYERS = J["results"], J["layers"]
sizes = list(range(1, len(LAYERS) + 1))
ks = sizes[1:]
colors = {"RAEv2": "gray", "p0.9 (drop0.9)": "#F6850C"}
models = [m for m in colors if m in res]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for m in models:
    ax1.plot(ks, res[m]["marg_db"], "o-", ms=3, color=colors[m], label=m)
ax1.set_title("Marginal  d(k) = v(k) - v(k-1)  [dB]")
ax1.set_xlabel("k-th layer added"); ax1.set_ylabel("dPSNR [dB]")
ax1.grid(alpha=0.3); ax1.legend()

for m in models:
    ax2.plot(ks, res[m]["marg_mse"], "o-", ms=3, color=colors[m], label=m)
ax2.set_title("Marginal  MSE reduction")
ax2.set_xlabel("k-th layer added"); ax2.set_ylabel("MSE(k-1) - MSE(k)")
ax2.grid(alpha=0.3); ax2.legend()
ax2.yaxis.tick_right(); ax2.yaxis.set_label_position("right")

fig.tight_layout(); fig.subplots_adjust(wspace=0.06)
for ext in ("png", "pdf"):
    fig.savefig(f"output_p09/subset_sweep_marginal.{ext}", dpi=130, bbox_inches="tight")
print("saved -> output_p09/subset_sweep_marginal.png")
