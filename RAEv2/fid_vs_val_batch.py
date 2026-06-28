#!/usr/bin/env python3
"""Batch gFID+IS vs the ImageNet val 50k reference.

Loads the 50k-val npz reference ONCE into RAM, then computes FID+IS for every
generated .npy passed (each [N,256,256,3] uint8), writing one JSON per gen file.
Reuses the validated calculate_fid_isc (torch-fidelity) so numbers match the
rest of the pipeline. Run sequentially on one GPU after generation finishes.

Usage:
  python fid_vs_val_batch.py --ref <val.npz> --gens a.npy b.npy ... --outdir <dir>
  python fid_vs_val_batch.py --ref <val.npz> --glob "<dir>/*.npy" --outdir <dir>
"""
import argparse
import glob as globmod
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from eval.fid import calculate_fid_isc

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True, help="reference npz (key arr_0), e.g. imagenet-256-val 50k")
ap.add_argument("--gens", nargs="*", default=[], help="gen .npy paths")
ap.add_argument("--glob", default=None, help="glob for gen .npy paths")
ap.add_argument("--outdir", required=True)
ap.add_argument("--bs", type=int, default=64)
a = ap.parse_args()

gens = list(a.gens)
if a.glob:
    gens += sorted(globmod.glob(a.glob))
gens = sorted(set(gens))
assert gens, "no gen npys given"

_z = np.load(a.ref)
ref = _z["arr_0"] if "arr_0" in _z else _z[list(_z.keys())[0]]
print(f"loaded reference {ref.shape} {ref.dtype} from {a.ref}", flush=True)
os.makedirs(a.outdir, exist_ok=True)

results = []
for gp in gens:
    name = os.path.splitext(os.path.basename(gp))[0]
    outj = os.path.join(a.outdir, name + ".json")
    gen = np.load(gp)
    fid, isc = calculate_fid_isc(gen, ref, bs=a.bs, device="cuda")
    rec = {"gen": gp, "name": name, "fid": float(fid), "is": float(isc),
           "num_gen": int(gen.shape[0]), "ref_num": int(ref.shape[0])}
    json.dump(rec, open(outj, "w"), indent=2)
    results.append(rec)
    print(f"  {name}: FID={fid:.3f} IS={isc:.2f} ({gen.shape[0]} gen vs {ref.shape[0]} ref) -> {outj}", flush=True)

print("\n=== summary (sorted by FID) ===", flush=True)
for r in sorted(results, key=lambda r: r["fid"]):
    print(f"  {r['name']:36s} FID={r['fid']:7.3f}  IS={r['is']:7.2f}", flush=True)
print("FID BATCH DONE", flush=True)
