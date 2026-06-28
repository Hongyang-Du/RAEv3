#!/usr/bin/env python3
"""Generate several guidance combos in ONE process, loading each (config,ckpt) only
once (model load ~60-90s dominates otherwise). Used to run the IG x CFG sweep with
full GPU utilization: one gen_multi.py per GPU, each handling a slice of the combos.

Jobs file: one job per line, TAB-separated:
    config <TAB> ckpt <TAB> ig <TAB> cfg <TAB> n <TAB> steps <TAB> out
Jobs are sorted by (config,ckpt) so the model is reused across same-checkpoint combos.
"""
import argparse
import dataclasses
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import numpy as np
import torch
from omegaconf import OmegaConf

from configs.stage2 import CFGConfig, GuidanceConfig, IGConfig, Stage2Config
import eval_fid_dit as efd
from stage2.transport import create_sampler, create_transport
from utils.guidance_utils import get_model_forward_fn
from utils.model_utils import instantiate_from_config


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--jobs", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--label-offset", type=int, default=0,
                help="add to the class index (for class-balanced sharding: shard g uses offset g*shard_size)")
args = ap.parse_args()

jobs = []
for ln in open(args.jobs):
    ln = ln.strip()
    if not ln or ln.startswith("#"):
        continue
    cfg, ckpt, ig, cfgs, n, steps, out = ln.split("\t")
    jobs.append(dict(config=cfg, ckpt=ckpt, ig=float(ig), cfg=float(cfgs),
                     n=int(n), steps=int(steps), out=out))
# pending only, sorted so identical (config,ckpt) are contiguous
jobs = [j for j in jobs if not os.path.exists(j["out"])]
jobs.sort(key=lambda j: (j["config"], j["ckpt"]))
log(f"{len(jobs)} pending jobs on this GPU")

cur_key = None
rae = model = transport = config = None
for j in jobs:
    key = (j["config"], j["ckpt"])
    if key != cur_key:
        del model, rae
        torch.cuda.empty_cache()
        config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(j["config"])))
        config.post_process(); config.prepare_model_params()
        rae = instantiate_from_config(config.stage_1).to("cuda").eval()
        model = instantiate_from_config(config.stage_2).to("cuda").eval()
        ck = torch.load(j["ckpt"], map_location="cpu", weights_only=False)
        model.load_state_dict(efd._strip_prefixes(ck["model"])); del ck
        ls = tuple(config.misc.latent_size)
        tds = math.sqrt((config.misc.time_dist_shift_dim or math.prod(ls)) / config.misc.time_dist_shift_base)
        transport = create_transport(config=config.transport, time_dist_shift=tds)
        cur_key = key
        log(f"loaded model for {os.path.basename(j['ckpt'])} ({os.path.basename(j['config'])})")

    ls = tuple(config.misc.latent_size)
    null_label = config.misc.num_classes
    guid = GuidanceConfig(cfg=CFGConfig(scale=j["cfg"], t_min=0.0, t_max=1.0),
                          ig=IGConfig(scale=j["ig"], t_min=0.10, t_max=1.0))
    model_fn, skw = get_model_forward_fn(model, guid)
    use_g = guid.any_guidance_active
    sampler = create_sampler(transport, guidance_config=guid)
    ode = sampler.sample_ode(**{**dataclasses.asdict(config.sampler), "num_steps": j["steps"]})
    g = torch.Generator(device="cuda").manual_seed(args.seed)
    arr = np.empty((j["n"], config.training.image_size, config.training.image_size, 3), dtype=np.uint8)
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, j["n"], args.batch):
            y = ((args.label_offset + torch.arange(i, min(i + args.batch, j["n"]))) % config.misc.num_classes).to("cuda")
            bs = len(y)
            zs = torch.randn(bs, *ls, device="cuda", generator=g)
            if use_g:
                zs = torch.cat([zs, zs], dim=0)
                y = torch.cat([y, torch.full((bs,), null_label, device="cuda", dtype=y.dtype)], dim=0)
            lat = ode(zs, model_fn, context=y, attn_mask=None, **skw)[-1]
            if use_g:
                lat = lat.chunk(2, dim=0)[0]
            imgs = rae.decode(lat.float()).clamp(0, 1)
            arr[i:i + bs] = (imgs * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    tmp = j["out"] + ".tmp.npy"
    np.save(tmp, arr); os.replace(tmp, j["out"])
    log(f"DONE ig={j['ig']} cfg={j['cfg']} n={j['n']} in {time.time()-t0:.0f}s -> {os.path.basename(j['out'])}")

log("GPU JOBS DONE")
