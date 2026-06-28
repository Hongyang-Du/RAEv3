#!/usr/bin/env python3
"""Tabulate the IG x CFG gFID sweep.

Reads per-combo JSONs ({name: '<exp>_ig<IG>_cfg<CFG>_ep<EP>', fid, is}) and prints,
per exp, a gFID grid (rows=IG, cols=CFG) + IS grid, the best combo, and the delta
vs the (ig=1.0, cfg=1.0) no-guidance baseline on the same 50k-val reference.

Usage: python tabulate_gfid_guidance.py [results_dir]
"""
import glob
import json
import os
import re
import sys

RESDIR = sys.argv[1] if len(sys.argv) > 1 else "/sensei-fs-3/users/hongyangd/ckpt/gfid_guidance/fid"
PAT = re.compile(r"(?P<exp>.+?)_ig(?P<ig>[0-9.]+)_cfg(?P<cfg>[0-9.]+)_ep(?P<ep>\d+)")

rows = {}  # exp -> {(ig,cfg): (fid, is)}
for f in sorted(glob.glob(os.path.join(RESDIR, "*.json"))):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    m = PAT.match(d.get("name", os.path.basename(f)[:-5]))
    if not m:
        continue
    exp = m.group("exp")
    ig = float(m.group("ig")); cfg = float(m.group("cfg"))
    rows.setdefault(exp, {})[(ig, cfg)] = (d.get("fid"), d.get("is"), d.get("num_gen"))

for exp in sorted(rows):
    cell = rows[exp]
    igs = sorted({k[0] for k in cell})
    cfgs = sorted({k[1] for k in cell})
    ng = next((v[2] for v in cell.values() if v[2]), "?")
    print(f"\n{'='*68}\n{exp}  (ep80, {ng}-gen vs 50k-val)\n{'='*68}")
    # gFID grid
    print("gFID ↓     " + "".join(f"cfg={c:<7}" for c in cfgs))
    for ig in igs:
        line = f"  ig={ig:<5} "
        for c in cfgs:
            v = cell.get((ig, c))
            line += f"{v[0]:<11.3f}" if v and v[0] is not None else f"{'-':<11}"
        print(line)
    # IS grid
    print("IS ↑       " + "".join(f"cfg={c:<7}" for c in cfgs))
    for ig in igs:
        line = f"  ig={ig:<5} "
        for c in cfgs:
            v = cell.get((ig, c))
            line += f"{v[1]:<11.2f}" if v and v[1] is not None else f"{'-':<11}"
        print(line)
    # best + delta vs baseline
    valid = {k: v for k, v in cell.items() if v[0] is not None}
    if not valid:
        continue
    best = min(valid, key=lambda k: valid[k][0])
    base = cell.get((1.0, 1.0))
    bf = valid[best][0]
    print(f"  best: ig={best[0]} cfg={best[1]} -> gFID {bf:.3f} (IS {valid[best][1]:.2f})")
    if base and base[0] is not None:
        print(f"  no-guidance (ig=1,cfg=1): gFID {base[0]:.3f}  |  best improves by {base[0]-bf:.3f} "
              f"({100*(base[0]-bf)/base[0]:.1f}%)")
