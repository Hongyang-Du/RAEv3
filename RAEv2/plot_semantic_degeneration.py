#!/usr/bin/env python3
"""Semantic-degeneration figure from output_full/semantic_probe/results.json.

One continuous depth axis: DINOv3 encoder layers (semantics RISE) flow into the
stage-1 decoder transformer blocks (semantics FALL as the decoder trades semantics
for pixels). Three decoder curves: ours (16ep k23), official RAEv2 k7, official k23.

Design choices for readability:
  - shared x-axis; encoder region shaded, decoder region white, divider line
  - encoder one curve; decoders three curves with distinct colors/markers
  - a connector from the encoder's last point to each decoder's first block
  - annotated peak (encoder) and the decoder drop
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

RES = "output_full/semantic_probe/results.json"
OUT = "output_full/semantic_degeneration.png"

res = json.load(open(RES))
acc = res["history"][-1]
epoch = acc["epoch"]
enc_layers = res["enc_probe_layers"]
block_ids = res["block_ids"]
n_blocks = res["n_blocks"]

# --- build a single monotone x-axis: encoder layers, then decoder blocks ---
# encoder occupies x = 0 .. (E-1); decoder blocks placed right after with a small gap.
E = len(enc_layers)
GAP = 1.0
enc_x = list(range(E))
dec_x = [E - 1 + GAP + (b / max(block_ids)) * (E * 1.6) for b in block_ids]  # spread decoder over a wide span

enc_y = [acc[f"enc_L{l}"] for l in enc_layers]
ours_y = [acc[f"ours_b{b}"] for b in block_ids]
k7_y = [acc[f"k7_b{b}"] for b in block_ids]
k23_y = [acc[f"k23_b{b}"] for b in block_ids]

plt.rcParams.update({
    "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150,
})
fig, ax = plt.subplots(figsize=(11, 6))

# ours = orange (the highlight); RAEv2 baselines = grays (dark/light); encoder = slate purple.
ENC_C = "#5b507a"      # slate purple (encoder)
OURS_C = "#f58426"     # orange        (ours, highlighted)
K23_C = "#4d4d4d"      # dark gray     (raev2 k23)
K7_C = "#9e9e9e"       # light gray    (raev2 k7)

# region shading + divider
div = E - 1 + GAP / 2
ax.axvspan(min(enc_x) - 0.4, div, color="#dfe7f5", alpha=0.45, lw=0, zorder=0)
ax.axvspan(div, max(dec_x) + 0.6, color="#f7f7f7", alpha=0.7, lw=0, zorder=0)
ax.axvline(div, color="0.5", ls=":", lw=1.2, zorder=1)

# encoder curve (rising)
ax.plot(enc_x, enc_y, "-o", color=ENC_C, lw=2.8, ms=7, zorder=5)

# connectors encoder-last -> each decoder first block (dashed, faint, curve color)
for dy, c in ((ours_y, OURS_C), (k7_y, K7_C), (k23_y, K23_C)):
    ax.plot([enc_x[-1], dec_x[0]], [enc_y[-1], dy[0]], "--", color=c, lw=1.2, alpha=0.55, zorder=3)

# decoder curves (falling) — ours highlighted (orange, thick), RAEv2 grays
ax.plot(dec_x, k23_y, "-^", color=K23_C, lw=2.2, ms=6, zorder=4)
ax.plot(dec_x, k7_y, "-s", color=K7_C, lw=2.2, ms=6, zorder=4)
ax.plot(dec_x, ours_y, "-o", color=OURS_C, lw=3.0, ms=8, zorder=6)

# headroom on top so nothing collides with the encoder peak
ymin = min(min(enc_y), min(ours_y), min(k7_y), min(k23_y))
ymax = max(max(enc_y), max(ours_y), max(k7_y), max(k23_y))
ax.set_ylim(ymin - 4, ymax + 12)

# --- in-figure labels at the end of each curve (no legend) ---
# encoder label sits to the upper-left of its peak (away from the curve)
ax.annotate("DINOv3 encoder", (enc_x[-1], enc_y[-1]), textcoords="offset points",
            xytext=(-10, 8), ha="right", va="bottom", color=ENC_C, fontsize=13, fontweight="bold")
# decoder end labels: explicit, vertically separated y-positions to avoid overlap
xr = dec_x[-1]
xlab = xr + (max(dec_x) - min(enc_x)) * 0.025
# three curves all converge near ~55%; fan the labels out to +6 / 0 / -6 in y
ax.annotate("ours (OmniRAE)", (xr, ours_y[-1]), xytext=(xlab, ours_y[-1] + 6.0),
            ha="left", va="center", color=OURS_C, fontsize=13, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=OURS_C, lw=1.0, alpha=0.6))
ax.annotate("RAEv2 K=23", (xr, k23_y[-1]), xytext=(xlab, k23_y[-1] + 0.5),
            ha="left", va="center", color=K23_C, fontsize=11,
            arrowprops=dict(arrowstyle="-", color=K23_C, lw=0.9, alpha=0.5))
ax.annotate("RAEv2 K=7", (dec_x[-1], k7_y[-1]), xytext=(xlab, k7_y[-1] - 5.0),
            ha="left", va="center", color=K7_C, fontsize=11,
            arrowprops=dict(arrowstyle="-", color=K7_C, lw=0.9, alpha=0.5))

# region labels — placed just under the top of the new headroom
ytxt = ymax + 9
ax.text((min(enc_x) + div) / 2, ytxt, "encoder — semantics build up",
        ha="center", va="center", fontsize=11.5, color=ENC_C, alpha=0.9, style="italic")
ax.text((div + max(dec_x)) / 2, ytxt, "decoder — semantics degenerate",
        ha="center", va="center", fontsize=11.5, color="0.4", style="italic")

# x ticks: encoder layer labels + a few decoder block labels
xticks = enc_x + [dec_x[i] for i in range(0, len(block_ids), 2)]
xlabels = [f"L{l}" for l in enc_layers] + [f"b{block_ids[i]}" for i in range(0, len(block_ids), 2)]
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, fontsize=9)

# extra right margin so the end labels fit
ax.set_xlim(min(enc_x) - 0.6, max(dec_x) + (max(dec_x) - min(enc_x)) * 0.22)
ax.set_xlabel("depth  —  encoder layer  →  decoder block", fontsize=12)
ax.set_ylabel("ImageNet val top-1 [%]  (linear probe on cls token)", fontsize=12)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"saved -> {OUT}")

# also print the table of numbers
print(f"\n[epoch {epoch}] encoder:", {f"L{l}": round(acc[f'enc_L{l}'], 1) for l in enc_layers})
print("ours  :", {f"b{b}": round(acc[f'ours_b{b}'], 1) for b in block_ids})
print("k7    :", {f"b{b}": round(acc[f'k7_b{b}'], 1) for b in block_ids})
print("k23   :", {f"b{b}": round(acc[f'k23_b{b}'], 1) for b in block_ids})
