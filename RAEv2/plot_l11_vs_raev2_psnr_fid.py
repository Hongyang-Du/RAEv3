#!/usr/bin/env python3
"""L11 (single shallowest layer) vs raev2 K=7: decoder reconstruction PSNR per
epoch (left axis, higher better) + final stage-2 DiT generation FID (right axis,
lower better; horizontal line + star + value). Reconstruction<->generation trade-off."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EP = [1, 2, 3, 4, 5]
L11_PSNR   = [21.08, 25.55, 27.93, 28.97, 29.25]
RAEV2_PSNR = [18.63, 23.30, 25.04, 25.87, 25.98]
L11_FID, RAEV2_FID = 23.13, 16.47          # stage-2 DiT gFID (epoch 10)

C_L11, C_RAEV2 = "#36ADA3", "#9FA1FF"

fig, ax1 = plt.subplots(figsize=(9, 5.5))
ax2 = ax1.twinx()

# left axis: decoder reconstruction PSNR (solid, per epoch) -- legend = color only
ax1.plot(EP, L11_PSNR,   "-o", color=C_L11,   lw=2.6, mec="k", mew=0.5, ms=7, label="L11")
ax1.plot(EP, RAEV2_PSNR, "-o", color=C_RAEV2, lw=2.6, mec="k", mew=0.5, ms=7, label="raev2 K=7")

# right axis: final DiT gFID as horizontal line + star + value (value to the
# right of the star so neither the line nor the marker covers it)
for fid, c in [(L11_FID, C_L11), (RAEV2_FID, C_RAEV2)]:
    ax2.axhline(fid, color=c, ls="--", lw=2.2)
    ax2.plot(5.05, fid, "*", color=c, ms=22, mec="k", mew=0.7)
    ax2.annotate(f"{fid:.2f}", (5.05, fid), textcoords="offset points", xytext=(16, 0),
                 ha="left", va="center", color="k", fontsize=11, fontweight="bold")

ax1.set_xlabel("epoch")
ax1.set_ylabel("decoder reconstruction PSNR [dB]  (higher = better)")
ax2.set_ylabel("DiT generation FID  (lower = better)")
ax1.set_xlim(0.7, 5.9)
ax1.set_xticks(EP)
ax2.set_ylim(0, 60)                      # large scale -> the two FID lines sit close together
ax1.grid(alpha=0.3)

ax1.legend(title="color = model", fontsize=10, loc="lower right")
fig.tight_layout()
for ext in ("pdf", "png"):
    out = f"output_full/l11_vs_raev2_psnr_fid.{ext}"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}")
