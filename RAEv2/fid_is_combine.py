#!/usr/bin/env python3
"""Combine generated shards + a shared real reference -> FID + IS (one torch-fidelity pass)."""
import argparse
import json
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from omegaconf import OmegaConf
from configs.stage2 import Stage2Config
import eval_fid_dit as efd
from eval.fid import calculate_fid_isc

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--shards", required=True, help="comma-sep .npy paths")
ap.add_argument("--data", default="./data/imagenet-256")
ap.add_argument("--total", type=int, default=10000)
ap.add_argument("--ref-num", type=int, default=10000)
ap.add_argument("--ref-seed", type=int, default=42)
ap.add_argument("--ref-npy", default=None, help="cache path for the real reference set")
ap.add_argument("--ref-npz", default=None, help="If set, use this npz (key arr_0) as the reference (e.g. imagenet-256-val 50k). Overrides --ref-npy/--data.")
ap.add_argument("--num-workers", type=int, default=8)
ap.add_argument("--epoch", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()

config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(a.config)))
config.post_process()

gen = np.concatenate([np.load(s) for s in a.shards.split(",")], axis=0)[:a.total]
print(f"gen total {gen.shape}", flush=True)

if a.ref_npz:
    _z = np.load(a.ref_npz)
    ref = _z["arr_0"] if "arr_0" in _z else _z[list(_z.keys())[0]]
    print(f"loaded reference {ref.shape} from {a.ref_npz} (npz)", flush=True)
elif a.ref_npy and os.path.exists(a.ref_npy):
    ref = np.load(a.ref_npy)
    print(f"loaded cached reference {ref.shape} from {a.ref_npy}", flush=True)
else:
    rargs = types.SimpleNamespace(data=a.data, seed=a.ref_seed, num_samples=a.ref_num, num_workers=a.num_workers)
    ref = efd.load_reference(rargs, config.training.image_size)
    if a.ref_npy:
        np.save(a.ref_npy, ref)

fid, isc = calculate_fid_isc(gen, ref, bs=64, device="cuda")
result = {"fid": fid, "is": isc, "epoch": a.epoch, "num_samples": int(gen.shape[0]), "ref_num": int(ref.shape[0])}
json.dump(result, open(a.out, "w"), indent=2)
print(f"FID = {fid:.3f}  IS = {isc:.3f}  ({gen.shape[0]} gen vs {ref.shape[0]} real) -> {a.out}", flush=True)
