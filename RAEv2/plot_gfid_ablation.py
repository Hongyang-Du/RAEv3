#!/usr/bin/env python3
"""Plot gFID + IS convergence curves from ckpt/gfid_ablation/<exp>_ep<E>.json.
Usage: python plot_gfid_ablation.py [results_dir]
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESDIR = sys.argv[1] if len(sys.argv) > 1 else "/sensei-fs-3/users/hongyangd/ckpt/gfid_ablation"
LABEL = {"h1": "h1 (plain dec)", "encoder": "encoder (plain dec)", "sigreg": "sigreg (sigreg dec)"}
COLOR = {"h1": "tab:blue", "encoder": "tab:green", "sigreg": "tab:red"}

data = {}  # exp -> list of (epoch, fid, is)
for f in glob.glob(os.path.join(RESDIR, "*_ep*.json")):
    m = re.match(r"(.+)_ep(\d+)\.json", os.path.basename(f))
    if not m:
        continue
    exp, ep = m.group(1), int(m.group(2))
    try:
        d = json.load(open(f))
    except Exception:
        continue
    data.setdefault(exp, []).append((ep, d.get("fid"), d.get("is")))

for exp in data:
    data[exp].sort()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for exp, rows in sorted(data.items()):
    eps = [r[0] for r in rows]
    fids = [r[1] for r in rows]
    iss = [r[2] for r in rows]
    c, lab = COLOR.get(exp, None), LABEL.get(exp, exp)
    ax1.plot(eps, fids, "o-", color=c, label=lab)
    ax2.plot(eps, iss, "o-", color=c, label=lab)
    for e, fv in zip(eps, fids):
        if fv is not None:
            ax1.annotate(f"{fv:.1f}", (e, fv), fontsize=7, textcoords="offset points", xytext=(0, 5))

ax1.set_xlabel("epoch"); ax1.set_ylabel("gFID ↓"); ax1.set_title("gFID vs epoch"); ax1.grid(alpha=.3); ax1.legend()
ax2.set_xlabel("epoch"); ax2.set_ylabel("IS ↑"); ax2.set_title("Inception Score vs epoch"); ax2.grid(alpha=.3); ax2.legend()
fig.suptitle("DiT convergence ablation (every-10-epoch ckpts, plain-conditional sampling)")
fig.tight_layout()
out = os.path.join(RESDIR, "gfid_is_curves.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved -> {out}")

# also dump a text table
print("\nexp           epoch   gFID     IS")
for exp, rows in sorted(data.items()):
    for ep, fv, iv in rows:
        print(f"{exp:12s}  {ep:5d}  {fv if fv is None else round(fv,2):>7}  {iv if iv is None else round(iv,2):>6}")
