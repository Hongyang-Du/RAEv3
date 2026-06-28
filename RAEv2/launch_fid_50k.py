#!/usr/bin/env python3
"""Parallel combine+FID for the 50k grid: distribute combos across GPUs, each concatenates
its combos' 8 shards and computes gFID+IS vs the val 50k npz."""
import glob
import os
import re
import subprocess

ROOT = "/sensei-fs-3/users/hongyangd/RAEv3/RAEv2"
OUT = os.environ.get("GFID_OUT", "/opt/cis/hongyangd_gfid")
GEN = os.path.join(OUT, "gen")
# FID JSONs are tiny -> keep them in the sensei folder (results), shards stay on local scratch.
FID = os.environ.get("GFID_FID", "/sensei-fs-3/users/hongyangd/ckpt/gfid_guidance_50k/fid")
REF = os.environ.get("REF", "/sensei-fs-3/users/hongyangd/official_raev2/data/imagenet-256/imagenet-256-val.npz")
NGPU = int(os.environ.get("NGPU", "8"))
os.makedirs(FID, exist_ok=True)

# discover combo prefixes from shard files
prefixes = sorted({re.sub(r"_s\d+\.npy$", "", os.path.basename(p))
                   for p in glob.glob(os.path.join(GEN, "*_s*.npy"))})
todo = [p for p in prefixes if not os.path.exists(os.path.join(FID, p + ".json"))]
print(f"{len(prefixes)} combos found, {len(todo)} need FID", flush=True)
if not todo:
    raise SystemExit(0)

buckets = [[] for _ in range(NGPU)]
for i, p in enumerate(todo):
    buckets[i % NGPU].append(p)

procs = []
for g, bucket in enumerate(buckets):
    if not bucket:
        continue
    logf = open(os.path.join(FID, f"fid_gpu{g}.log"), "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
    cmd = ["/sensei-fs-3/users/hongyangd/rae_env/bin/python", "-u",
           os.path.join(ROOT, "fid_50k_batch.py"), "--ref", REF, "--gendir", GEN,
           "--outdir", FID, "--combos"] + bucket
    procs.append((g, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)))
    print(f"GPU{g}: {bucket}", flush=True)

rc = 0
for g, p in procs:
    r = p.wait(); rc = rc or r
    print(f"GPU{g} fid rc={r}", flush=True)
print(f"ALL 50k FID DONE rc={rc}", flush=True)
