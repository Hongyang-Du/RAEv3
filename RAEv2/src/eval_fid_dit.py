#!/usr/bin/env python3
"""Offline generation FID for a trained stage-2 DiT checkpoint.

Samples N class-conditional images (classes cycled 0..999 evenly, fixed seed,
same Euler ODE sampler as training-time viz / sample_dit.py), decodes with the
stage-1 decoder, and computes FID (torch-fidelity) against N real ImageNet
training images (Resize256+CenterCrop256 — same preprocessing as stage-2
training data). Single GPU, ~30-45 min for 10k samples.

Latent-space losses are not comparable across stage-1 variants; pixel-space
FID with a shared reference set IS — use the same --num-samples/--seed for
every run you compare.

Usage (inside rae container, one free GPU):
    python src/eval_fid_dit.py \
        --config configs/stage2/training/imagenet-dinov3l-l11-raev2mls.yaml \
        --ckpt   ckpts_full/stage2/dit-l11/checkpoints/ep-0000009.pt \
        --data   /datasets/imagenet-256-full \
        --num-samples 10000 \
        --out    ckpts_full/stage2/dit-l11/fid.json
"""

import argparse
import dataclasses
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.utils import save_image

from configs.stage2 import Stage2Config
from eval.fid import calculate_fid_isc
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


def _to_uint8_nhwc(imgs01: torch.Tensor) -> np.ndarray:
    """[B,3,H,W] float in [0,1] -> [B,H,W,3] uint8."""
    return (imgs01.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def _build_guidance(args):
    """Build a GuidanceConfig from optional args. Defaults (scale=1.0) => no guidance,
    so callers that don't set guidance args get the original plain-conditional path."""
    from configs.stage2 import CFGConfig, GuidanceConfig, IGConfig
    return GuidanceConfig(
        cfg=CFGConfig(scale=getattr(args, "cfg_scale", 1.0),
                      t_min=getattr(args, "cfg_tmin", 0.0),
                      t_max=getattr(args, "cfg_tmax", 1.0)),
        ig=IGConfig(scale=getattr(args, "ig_scale", 1.0),
                    t_min=getattr(args, "ig_tmin", 0.0),
                    t_max=getattr(args, "ig_tmax", 1.0),
                    unconditional_scale=getattr(args, "uncond_ig_scale", None)),
    )


def generate(args, config, device):
    from utils.guidance_utils import get_model_forward_fn
    rae = instantiate_from_config(config.stage_1).to(device).eval()
    model = instantiate_from_config(config.stage_2).to(device).eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = _strip_prefixes(ck["model" if args.raw else "ema"])
    model.load_state_dict(sd)
    epoch = ck.get("epoch", "?")
    del ck, sd
    print(f"Loaded {'raw' if args.raw else 'EMA'} DiT from {args.ckpt} (epoch {epoch})", flush=True)

    # Guidance: IG (base + s*(full-base)) and/or CFG (cond/uncond). When active, the
    # forward fns from guidance_utils expect a DOUBLED batch [cond; uncond] and return
    # it doubled; we take the first half before decoding (matches evaluate_generation_distributed).
    guid = _build_guidance(args)
    model_fn, sample_kwargs = get_model_forward_fn(model, guid)
    use_guidance = guid.any_guidance_active
    print(f"Guidance: use_guidance={use_guidance} | "
          f"IG(use={guid.use_ig}, scale={guid.ig.scale}, t=[{guid.ig.t_min},{guid.ig.t_max}], "
          f"uncond={guid.ig.unconditional_scale}) | "
          f"CFG(use={guid.use_cfg}, scale={guid.cfg.scale}, t=[{guid.cfg.t_min},{guid.cfg.t_max}])", flush=True)

    latent_size = tuple(config.misc.latent_size)
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size)) / config.misc.time_dist_shift_base)
    transport = create_transport(config=config.transport, time_dist_shift=time_dist_shift)
    sampler = create_sampler(transport, guidance_config=guid)
    ode = sampler.sample_ode(**{**dataclasses.asdict(config.sampler), "num_steps": args.steps})

    n = args.num_samples
    null_label = config.misc.num_classes
    labels = torch.arange(n) % config.misc.num_classes        # even class coverage
    g = torch.Generator(device=device).manual_seed(args.seed)
    arr = np.empty((n, config.training.image_size, config.training.image_size, 3), dtype=np.uint8)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, n, args.batch):
            y = labels[i:i + args.batch].to(device)
            bs = len(y)
            zs = torch.randn(bs, *latent_size, device=device, generator=g)
            if use_guidance:
                zs = torch.cat([zs, zs], dim=0)
                y_null = torch.full((bs,), null_label, device=device, dtype=y.dtype)
                y_in = torch.cat([y, y_null], dim=0)
            else:
                y_in = y
            latents = ode(zs, model_fn, context=y_in, attn_mask=None, **sample_kwargs)[-1]
            if use_guidance:
                latents = latents.chunk(2, dim=0)[0]
            imgs = rae.decode(latents.float()).clamp(0, 1)
            arr[i:i + bs] = _to_uint8_nhwc(imgs)
            if (i // args.batch) % 10 == 0:
                print(f"  generated {i + bs}/{n}", flush=True)

    if args.grid:
        os.makedirs(os.path.dirname(os.path.abspath(args.grid)), exist_ok=True)
        grid = torch.from_numpy(arr[:64]).permute(0, 3, 1, 2).float() / 255
        save_image(grid, args.grid, nrow=8)
        print(f"Sample grid -> {args.grid}", flush=True)

    del model, rae
    torch.cuda.empty_cache()
    return arr, epoch


def load_reference(args, image_size):
    """N real training images with the stage-2 training preprocessing."""
    from data.partial_imagenet import PartialImageNetDataset
    tf = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    ds = PartialImageNetDataset(args.data, split="train", transform=tf)
    g = torch.Generator().manual_seed(args.seed)
    idxs = torch.randperm(len(ds), generator=g)[:args.num_samples].tolist()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idxs),
        batch_size=128, num_workers=args.num_workers, drop_last=False)

    arr = np.empty((len(idxs), image_size, image_size, 3), dtype=np.uint8)
    i = 0
    for imgs, _ in loader:
        arr[i:i + imgs.shape[0]] = _to_uint8_nhwc(imgs)
        i += imgs.shape[0]
        if (i // 128) % 20 == 0:
            print(f"  reference {i}/{len(idxs)}", flush=True)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      required=True, help="Stage-2 training YAML")
    ap.add_argument("--ckpt",        required=True, help="Stage-2 checkpoint (ep-*.pt)")
    ap.add_argument("--data",        default="/datasets/imagenet-256-full")
    ap.add_argument("--ref-npz",     default=None, help="If set, use this npz (key arr_0) as the reference (e.g. imagenet val 50k) instead of sampling train images")
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--batch",       type=int, default=64)
    ap.add_argument("--steps",       type=int, default=50)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--raw",         action="store_true", help="Use raw weights instead of EMA")
    ap.add_argument("--grid",        default=None, help="Optional PNG path for a 64-sample grid")
    ap.add_argument("--out",         default=None, help="Optional JSON path for the result")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device",      default="cuda:0")
    # Guidance (defaults => plain conditional, identical to the no-guidance baseline)
    ap.add_argument("--ig-scale",   type=float, default=1.0, help="Internal-guidance scale (1.0 = off)")
    ap.add_argument("--ig-tmin",    type=float, default=0.0)
    ap.add_argument("--ig-tmax",    type=float, default=1.0)
    ap.add_argument("--uncond-ig-scale", type=float, default=None, help="IG scale on the uncond branch (IG+CFG only)")
    ap.add_argument("--cfg-scale",  type=float, default=1.0, help="Classifier-free-guidance scale (1.0 = off)")
    ap.add_argument("--cfg-tmin",   type=float, default=0.0)
    ap.add_argument("--cfg-tmax",   type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device(args.device)
    config: Stage2Config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config)))
    config.post_process()
    config.prepare_model_params()
    # NOTE: this repo's simplified sampler applies NO guidance at inference —
    # the IG base head output is discarded (transport.get_drift takes tuple[0])
    # and there is no cond/uncond double forward. All FID numbers are
    # plain conditional Euler sampling.

    gen_arr, epoch = generate(args, config, device)
    if args.ref_npz:
        _z = np.load(args.ref_npz)
        ref_arr = _z["arr_0"] if "arr_0" in _z else _z[list(_z.keys())[0]]
        print(f"Loaded reference {ref_arr.shape} from {args.ref_npz}", flush=True)
    else:
        ref_arr = load_reference(args, config.training.image_size)

    print("Computing FID + IS (torch-fidelity)...", flush=True)
    fid, isc = calculate_fid_isc(gen_arr, ref_arr, bs=64, device="cuda")

    _guid = _build_guidance(args)
    result = {
        "fid": fid,
        "is": isc,
        "ckpt": args.ckpt,
        "epoch": epoch,
        "num_samples": args.num_samples,
        "steps": args.steps,
        "seed": args.seed,
        "ig_scale": args.ig_scale, "ig_tmin": args.ig_tmin, "ig_tmax": args.ig_tmax,
        "cfg_scale": args.cfg_scale, "cfg_tmin": args.cfg_tmin, "cfg_tmax": args.cfg_tmax,
        "uncond_ig_scale": args.uncond_ig_scale,
        "guidance": ("none (plain conditional Euler)" if not _guid.any_guidance_active
                     else f"IG={args.ig_scale}@[{args.ig_tmin},{args.ig_tmax}] CFG={args.cfg_scale}@[{args.cfg_tmin},{args.cfg_tmax}]"),
        "weights": "raw" if args.raw else "ema",
    }
    print(f"FID = {fid:.3f}  IS = {isc:.3f}  ({args.num_samples} gen vs {args.num_samples} real, "
          f"ckpt={args.ckpt}, epoch={epoch})", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
