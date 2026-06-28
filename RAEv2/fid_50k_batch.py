#!/usr/bin/env python3
"""Concatenate each combo's 8 shards -> 50k, compute gFID+IS vs the val 50k npz.

Loads the reference once. For each combo prefix, loads its shard npys (<prefix>_s*.npy),
concatenates to the full sample set, runs calculate_fid_isc, writes one JSON per combo.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from eval.fid import calculate_fid_isc

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--gendir", required=True)
ap.add_argument("--combos", nargs="+", required=True, help="combo prefixes (e.g. h1_ig1.0_cfg1.0_ep80)")
ap.add_argument("--outdir", required=True)
ap.add_argument("--bs", type=int, default=128)
a = ap.parse_args()

_z = np.load(a.ref)
ref = _z["arr_0"] if "arr_0" in _z else _z[list(_z.keys())[0]]
print(f"loaded reference {ref.shape} {ref.dtype}", flush=True)
os.makedirs(a.outdir, exist_ok=True)

for prefix in a.combos:
    outj = os.path.join(a.outdir, prefix + ".json")
    if os.path.exists(outj):
        print(f"skip {prefix} (done)", flush=True); continue
    shards = sorted(glob.glob(os.path.join(a.gendir, prefix + "_s*.npy")))
    if not shards:
        print(f"WARN no shards for {prefix}", flush=True); continue
    gen = np.concatenate([np.load(s) for s in shards], axis=0)
    fid, isc = calculate_fid_isc(gen, ref, bs=a.bs, device="cuda")
    rec = {"name": prefix, "fid": float(fid), "is": float(isc),
           "num_gen": int(gen.shape[0]), "ref_num": int(ref.shape[0]), "nshards": len(shards)}
    json.dump(rec, open(outj, "w"), indent=2)
    print(f"  {prefix}: FID={fid:.3f} IS={isc:.2f} ({gen.shape[0]} gen vs {ref.shape[0]} ref, {len(shards)} shards)", flush=True)
print("FID 50k BATCH DONE", flush=True)
