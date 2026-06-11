#!/usr/bin/env python3
"""Offline sampling from a trained stage-2 DiT checkpoint -> image grid PNG.

Loads the stage_1 wrapper + DiT (EMA weights) from a stage-2 training config/ckpt,
runs the same Euler ODE sampler training-time viz uses (no guidance), decodes, and
saves a grid. Works on any ep-*.pt while training is stopped or on a free GPU
(needs ~12 GB: DiT 3.5G + decoder + DINOv3-free... encoder is NOT loaded unless
--recon; plain sampling only needs DiT + decoder + stats).

Usage (inside rae container):
    python src/sample_dit.py \
        --config configs/stage2/training/imagenet-dinov3l-k7-raev2mls.yaml \
        --ckpt ckpts_full/stage2/dit-raev2mls-k7/checkpoints/ep-0000004.pt \
        --classes 207 360 387 974 88 979 417 279 \
        --out ckpts_full/stage2/dit-raev2mls-k7/samples/offline_ep4.png
"""

import argparse
import dataclasses
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

from configs.stage2 import Stage2Config
from stage2.transport import create_sampler, create_transport
from utils.model_utils import instantiate_from_config


def _strip_prefixes(sd):
    out = {}
    for k, v in sd.items():
        for p in ("module.", "_orig_mod."):
            while k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  required=True, help="Stage-2 training YAML")
    ap.add_argument("--ckpt",    required=True, help="Stage-2 checkpoint (ep-*.pt)")
    ap.add_argument("--classes", type=int, nargs="+", default=None,
                    help="ImageNet class ids; default: 16 spread over 0..999")
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--steps",   type=int, default=50)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--raw",     action="store_true", help="Use raw weights instead of EMA")
    ap.add_argument("--out",     required=True)
    ap.add_argument("--device",  default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    config: Stage2Config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config)))
    config.post_process()
    config.prepare_model_params()

    rae = instantiate_from_config(config.stage_1).to(device).eval()

    model = instantiate_from_config(config.stage_2).to(device).eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = _strip_prefixes(ck["model" if args.raw else "ema"])
    model.load_state_dict(sd)
    epoch = ck.get("epoch", "?")
    del ck, sd
    print(f"Loaded {'raw' if args.raw else 'EMA'} DiT from {args.ckpt} (epoch {epoch})")

    latent_size = tuple(config.misc.latent_size)
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size)) / config.misc.time_dist_shift_base)
    transport = create_transport(config=config.transport, time_dist_shift=time_dist_shift)
    sampler = create_sampler(transport, guidance_config=config.guidance)
    ode = sampler.sample_ode(**{**dataclasses.asdict(config.sampler), "num_steps": args.steps})

    classes = args.classes or [i * 999 // 15 for i in range(16)]
    y = torch.tensor([c for c in classes for _ in range(args.per_class)], device=device)
    n = y.shape[0]
    g = torch.Generator(device=device).manual_seed(args.seed)
    zs = torch.randn(n, *latent_size, device=device, generator=g)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        latents = ode(zs, model.forward, context=y, attn_mask=None)[-1]
        imgs = rae.decode(latents.float()).clamp(0, 1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_image(imgs.cpu(), args.out, nrow=round(n ** 0.5))
    print(f"{n} samples (classes {classes}, {args.steps} ODE steps) -> {args.out}")


if __name__ == "__main__":
    main()
