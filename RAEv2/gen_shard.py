#!/usr/bin/env python3
"""Generate one shard of class-conditional samples (reuses eval_fid_dit.generate)
and save as uint8 NHWC .npy. Used to parallelize a single gFID eval across GPUs."""
import argparse
import os
import sys
import types

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from configs.stage2 import Stage2Config
import eval_fid_dit as efd

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--num-samples", type=int, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--out", required=True)
# Guidance (defaults => plain conditional)
ap.add_argument("--ig-scale", type=float, default=1.0)
ap.add_argument("--ig-tmin", type=float, default=0.0)
ap.add_argument("--ig-tmax", type=float, default=1.0)
ap.add_argument("--uncond-ig-scale", type=float, default=None)
ap.add_argument("--cfg-scale", type=float, default=1.0)
ap.add_argument("--cfg-tmin", type=float, default=0.0)
ap.add_argument("--cfg-tmax", type=float, default=1.0)
a = ap.parse_args()

config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(a.config)))
config.post_process()
config.prepare_model_params()   # inject conditioning arch into stage_2 params (as eval_fid_dit.main does)

ga = types.SimpleNamespace(ckpt=a.ckpt, raw=True, num_samples=a.num_samples,
                           seed=a.seed, batch=a.batch, steps=a.steps, grid=None,
                           ig_scale=a.ig_scale, ig_tmin=a.ig_tmin, ig_tmax=a.ig_tmax,
                           uncond_ig_scale=a.uncond_ig_scale,
                           cfg_scale=a.cfg_scale, cfg_tmin=a.cfg_tmin, cfg_tmax=a.cfg_tmax)
arr, epoch = efd.generate(ga, config, "cuda")
np.save(a.out, arr)
print(f"shard saved {arr.shape} seed={a.seed} -> {a.out} (epoch {epoch})", flush=True)
