#!/usr/bin/env python3
"""Compute gFID+IS vs the val 50k npz for all generated combos, parallelized across GPUs.

Splits the gen .npy files across NGPU GPUs; each GPU runs fid_vs_val_batch.py on its
subset (loads the 50k ref once). With ~1TB RAM, 8 concurrent refs (~78GB) is fine.
"""
import glob
import os
import subprocess

ROOT = "/sensei-fs-3/users/hongyangd/RAEv3/RAEv2"
OUT = "/sensei-fs-3/users/hongyangd/ckpt/gfid_guidance"
GEN = os.path.join(OUT, "gen")
FID = os.path.join(OUT, "fid")
REF = os.environ.get("REF", "/sensei-fs-3/users/hongyangd/official_raev2/data/imagenet-256/imagenet-256-val.npz")
NGPU = int(os.environ.get("NGPU", "8"))
os.makedirs(FID, exist_ok=True)

gens = sorted(glob.glob(os.path.join(GEN, "*_ep80.npy")))
# skip combos already scored
todo = [g for g in gens if not os.path.exists(os.path.join(FID, os.path.basename(g)[:-4] + ".json"))]
print(f"{len(gens)} gen files, {len(todo)} need FID", flush=True)
if not todo:
    print("nothing to do", flush=True)
    raise SystemExit(0)

buckets = [[] for _ in range(NGPU)]
for i, g in enumerate(todo):
    buckets[i % NGPU].append(g)

procs = []
for gi, bucket in enumerate(buckets):
    if not bucket:
        continue
    logf = open(os.path.join(FID, f"fid_gpu{gi}.log"), "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi))
    cmd = ["/sensei-fs-3/users/hongyangd/rae_env/bin/python", "-u",
           os.path.join(ROOT, "fid_vs_val_batch.py"), "--ref", REF,
           "--outdir", FID, "--gens"] + bucket
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
    procs.append((gi, p))
    print(f"GPU{gi}: {len(bucket)} combos", flush=True)

rc = 0
for gi, p in procs:
    r = p.wait(); rc = rc or r
    print(f"GPU{gi} fid exited rc={r}", flush=True)
print(f"ALL FID DONE rc={rc}", flush=True)
