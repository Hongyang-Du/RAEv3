#!/usr/bin/env python3
"""Launch the IG x CFG gFID sweep across all 8 GPUs with load amortization.

Builds (exp, ig, cfg) jobs, cost-balances them into 8 sequential buckets (guided
jobs cost ~2x no-guidance) while keeping same-checkpoint jobs contiguous so each
GPU loads a model at most a couple times, then spawns one gen_multi.py per GPU.

Env: N (default 5000), STEPS (50), EP (80), EXPS ("h1 encoder sigreg"), NGPU (8).
Re-running is safe: gen_multi skips combos whose .npy already exists.
"""
import os
import subprocess
import sys

ROOT = "/sensei-fs-3/users/hongyangd/RAEv3/RAEv2"
OUT = "/sensei-fs-3/users/hongyangd/ckpt/gfid_guidance"
GEN = os.path.join(OUT, "gen")
JOBDIR = os.path.join(OUT, "jobs")
os.makedirs(GEN, exist_ok=True); os.makedirs(JOBDIR, exist_ok=True)

N = int(os.environ.get("N", "5000"))
STEPS = int(os.environ.get("STEPS", "50"))
EP = int(os.environ.get("EP", "80"))
NGPU = int(os.environ.get("NGPU", "8"))
EXPS = os.environ.get("EXPS", "h1 encoder sigreg").split()
EPT = f"{EP:07d}"

CFGY = {
    "h1": "configs/stage2/training/imagenet-dinov3l-h1decoder-plain-cls-k23.yaml",
    "encoder": "configs/stage2/training/imagenet-dinov3l-encoder-cls-k23.yaml",
    "sigreg": "configs/stage2/training/imagenet-dinov3l-sigreg-cls-k23.yaml",
}
DIR = {
    "h1": "/sensei-fs-3/users/hongyangd/ckpt/omnirae-dit-h1-plain-cls-k23-4node",
    "encoder": "/sensei-fs-3/users/hongyangd/ckpt/omnirae-dit-encoder-cls-k23-4node",
    "sigreg": "/sensei-fs-3/users/hongyangd/ckpt/omnirae-dit-sigreg-cls-k23-4node",
}
IGS = [1.0, 1.5, 2.0]
CFGS = [1.0, 1.5, 2.0]

# build jobs (skip exps whose ckpt is missing)
jobs = []
for exp in EXPS:
    ckpt = f"{DIR[exp]}/checkpoints/ep-{EPT}.pt"
    if not os.path.exists(ckpt):
        print(f"SKIP {exp}: missing {ckpt}", flush=True)
        continue
    for ig in IGS:
        for cfg in CFGS:
            out = f"{GEN}/{exp}_ig{ig}_cfg{cfg}_ep{EP}.npy"
            if os.path.exists(out):
                continue
            cost = 2 if (ig > 1.0 or cfg > 1.0) else 1
            jobs.append(dict(exp=exp, config=CFGY[exp], ckpt=ckpt, ig=ig, cfg=cfg,
                             n=N, steps=STEPS, out=out, cost=cost))

if not jobs:
    print("No pending jobs (all outputs exist or ckpts missing).", flush=True)
    sys.exit(0)

# keep same-ckpt jobs contiguous (for model reuse), then cost-balance into NGPU buckets
jobs.sort(key=lambda j: (j["exp"], j["ig"], j["cfg"]))
total = sum(j["cost"] for j in jobs)
target = total / NGPU
buckets = [[] for _ in range(NGPU)]
acc = 0.0; g = 0
for j in jobs:
    buckets[g].append(j)
    acc += j["cost"]
    if acc >= target * (g + 1) and g < NGPU - 1:
        g += 1
print(f"{len(jobs)} jobs, total cost {total}, ~{target:.1f}/gpu", flush=True)

procs = []
env_base = dict(os.environ,
                DINOV3_REPO_DIR="/sensei-fs-3/users/hongyangd/dinov3_repo",
                DINOV3_CKPT_DIR="/sensei-fs-3/users/hongyangd/pretrained_models/encoders/dinov3",
                TORCH_HOME="/sensei-fs-3/users/hongyangd/.cache/torch")
for gi, bucket in enumerate(buckets):
    if not bucket:
        continue
    jf = os.path.join(JOBDIR, f"jobs_gpu{gi}.txt")
    with open(jf, "w") as f:
        for j in bucket:
            f.write("\t".join([j["config"], j["ckpt"], str(j["ig"]), str(j["cfg"]),
                               str(j["n"]), str(j["steps"]), j["out"]]) + "\n")
    desc = " ".join(f"{j['exp']}({j['ig']},{j['cfg']})" for j in bucket)
    print(f"GPU{gi} ({sum(b['cost'] for b in bucket)} cost): {desc}", flush=True)
    logf = open(os.path.join(GEN, f"gpu{gi}.log"), "w")
    env = dict(env_base, CUDA_VISIBLE_DEVICES=str(gi))
    p = subprocess.Popen(["/sensei-fs-3/users/hongyangd/rae_env/bin/python", "-u",
                          os.path.join(ROOT, "gen_multi.py"), "--jobs", jf, "--batch", "64"],
                         cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
    procs.append((gi, p))

print(f"launched {len(procs)} GPU workers; waiting...", flush=True)
rc = 0
for gi, p in procs:
    r = p.wait()
    print(f"GPU{gi} exited rc={r}", flush=True)
    rc = rc or r
print(f"ALL WORKERS DONE rc={rc}", flush=True)
