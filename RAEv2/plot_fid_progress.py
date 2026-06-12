#!/usr/bin/env python3
"""FID-vs-epoch curves for all DiT variants, from run_fid_sweep.sh jsons
(ckpts_full/stage2/<run>/fid/fid_ep-*.json). Lower is better."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {
    "raev2 (MLS)":       ("ckpts_full/stage2/dit-raev2mls-k7", "tab:gray"),
    "nogate + SIGReg":   ("ckpts_full/stage2/dit-nogate-k7",   "tab:blue"),
    "dropmean + SIGReg": ("ckpts_full/stage2/dit-dropmean-k7", "tab:red"),
    "L11 only":          ("ckpts_full/stage2/dit-l11",         "tab:green"),
}

fig, ax = plt.subplots(figsize=(7, 5))
n_meta = None
for ri, (name, (expdir, color)) in enumerate(RUNS.items()):
    pts = []
    for f in sorted(glob.glob(os.path.join(expdir, "fid", "fid_ep-*.json"))):
        d = json.load(open(f))
        pts.append((int(d["epoch"]), d["fid"]))
        n_meta = (d["num_samples"], d["steps"])
    if not pts:
        continue
    pts.sort()
    ep, fid = zip(*pts)
    ax.plot(ep, fid, "o-", color=color, label=f"{name} (final {fid[-1]:.2f})")
    for e, f_ in zip(ep, fid):
        if e >= 4:                      # ep0/2 = EMA warmup noise, not annotated
            ax.annotate(f"{f_:.1f}", (e, f_), textcoords="offset points",
                        xytext=(0, 7 + 9 * ri), ha="center", fontsize=8, color=color)

ax.set_yscale("log")
ax.set_yticks([15, 20, 30, 50, 100, 200, 400])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("epoch (ckpt)")
ax.set_ylabel("FID, log scale  (lower = better)")
sub = f"{n_meta[0]} EMA samples vs real train images, {n_meta[1]}-step Euler, no guidance" if n_meta else ""
ax.set_title(f"Stage-2 DiT generation FID per checkpoint\n{sub}", fontsize=11)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
out = "ckpts_full/stage2/dit_fid_compare.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved -> {out}")
