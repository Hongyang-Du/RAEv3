#!/usr/bin/env python3
"""Full 50k-gen IG x CFG gFID grid, all 3 ckpts ep80, vs the val 50k npz.

Each combo's 50k samples are produced as 8 class-balanced shards (6250 each), one per
GPU. GPU g generates shard g of ALL 27 combos (seed 42+g, label offset g*6250); gen_multi
sorts by checkpoint so each GPU loads each of the 3 models once. All 8 GPUs run identical
balanced workloads with no barriers. Re-running resumes (gen_multi skips finished shards).

~14h. Then run combine+FID (concat 8 shards -> 50k -> FID vs ref).
Env: EP (80), NGPU (8), NTOTAL (50000).
"""
import math
import os
import subprocess

ROOT = "/sensei-fs-3/users/hongyangd/RAEv3/RAEv2"
# Shards go to node-local scratch (no sensei quota); only tiny FID JSONs land in the sensei folder.
OUT = os.environ.get("GFID_OUT", "/opt/cis/hongyangd_gfid")
GEN = os.path.join(OUT, "gen"); JOBDIR = os.path.join(OUT, "jobs")
os.makedirs(GEN, exist_ok=True); os.makedirs(JOBDIR, exist_ok=True)

EP = int(os.environ.get("EP", "80")); EPT = f"{EP:07d}"
NGPU = int(os.environ.get("NGPU", "8"))
NTOTAL = int(os.environ.get("NTOTAL", "50000"))
SHARD = math.ceil(NTOTAL / NGPU)
STEPS = 50
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
EXPS = os.environ.get("EXPS", "h1 encoder sigreg").split()
IGS = [1.0, 1.5, 2.0]; CFGS = [1.0, 1.5, 2.0]

combos = []
for exp in EXPS:
    ckpt = f"{DIR[exp]}/checkpoints/ep-{EPT}.pt"
    if not os.path.exists(ckpt):
        print(f"SKIP {exp}: missing {ckpt}", flush=True); continue
    for ig in IGS:
        for cfg in CFGS:
            combos.append((exp, CFGY[exp], ckpt, ig, cfg))
print(f"{len(combos)} combos x {NGPU} shards x {SHARD} = {len(combos)*NTOTAL} samples total "
      f"({NTOTAL}-gen/combo)", flush=True)

env_base = dict(os.environ,
                DINOV3_REPO_DIR="/sensei-fs-3/users/hongyangd/dinov3_repo",
                DINOV3_CKPT_DIR="/sensei-fs-3/users/hongyangd/pretrained_models/encoders/dinov3",
                TORCH_HOME="/sensei-fs-3/users/hongyangd/.cache/torch")
procs = []
for g in range(NGPU):
    jf = os.path.join(JOBDIR, f"jobs_gpu{g}.txt")
    with open(jf, "w") as f:
        for exp, cfgp, ckpt, ig, cfg in combos:
            out = f"{GEN}/{exp}_ig{ig}_cfg{cfg}_ep{EP}_s{g}.npy"
            f.write("\t".join([cfgp, ckpt, str(ig), str(cfg), str(SHARD), str(STEPS), out]) + "\n")
    logf = open(os.path.join(GEN, f"gpu{g}.log"), "w")
    env = dict(env_base, CUDA_VISIBLE_DEVICES=str(g))
    p = subprocess.Popen(["/sensei-fs-3/users/hongyangd/rae_env/bin/python", "-u",
                          os.path.join(ROOT, "gen_multi.py"), "--jobs", jf,
                          "--batch", "64", "--seed", str(42 + g), "--label-offset", str(g * SHARD)],
                         cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
    procs.append((g, p))
    print(f"GPU{g}: shard {g} of {len(combos)} combos (seed {42+g}, label_offset {g*SHARD})", flush=True)

print(f"launched {len(procs)} GPU workers; waiting (~14h)...", flush=True)
rc = 0
for g, p in procs:
    r = p.wait(); rc = rc or r
    print(f"GPU{g} exited rc={r}", flush=True)
print(f"ALL 50k GENERATION DONE rc={rc}", flush=True)
