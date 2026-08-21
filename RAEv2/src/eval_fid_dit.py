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
from eval.fid import calculate_rfid
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


def generate(args, config, device):
    rae = instantiate_from_config(config.stage_1).to(device).eval()
    model = instantiate_from_config(config.stage_2).to(device).eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = _strip_prefixes(ck["model" if args.raw else "ema"])
    model.load_state_dict(sd)
    epoch = ck.get("epoch", "?")
    del ck, sd
    print(f"Loaded {'raw' if args.raw else 'EMA'} DiT from {args.ckpt} (epoch {epoch})", flush=True)

    latent_size = tuple(config.misc.latent_size)
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size)) / config.misc.time_dist_shift_base)
    transport = create_transport(config=config.transport, time_dist_shift=time_dist_shift)
    sampler = create_sampler(transport, guidance_config=config.guidance)
    ode = sampler.sample_ode(**{**dataclasses.asdict(config.sampler), "num_steps": args.steps})

    # Standard classifier-free guidance. The repo's sampler/drift discards x_base and
    # ignores the omega token (use_cfg_conds=False), so guidance is implemented HERE:
    # a cond + uncond double-forward, combined at the model-prediction (x-pred) level
    #   guided = uncond + scale * (cond - uncond)
    # The uncond pass uses the dedicated null class (index = num_classes), which the
    # model was trained to recognise via 10% cfg-dropout. scale=1.0 (or None) reduces
    # to plain conditional sampling. The drift converts guided x-pred -> velocity.
    num_classes = config.misc.num_classes
    cfg_scale = args.cfg_scale
    if cfg_scale and cfg_scale != 1.0:
        null_label = torch.full((1,), num_classes, device=device, dtype=torch.long)

        def model_fn(x, t, **kw):
            ctx = kw["context"]
            attn = kw.get("attn_mask")
            null_ctx = null_label.expand(ctx.shape[0])
            out_c = model.forward(x, t, context=ctx, attn_mask=attn)
            out_u = model.forward(x, t, context=null_ctx, attn_mask=attn)
            oc = out_c[0] if isinstance(out_c, tuple) else out_c
            ou = out_u[0] if isinstance(out_u, tuple) else out_u
            return ou + cfg_scale * (oc - ou)
        print(f"CFG enabled: standard cond/uncond double-forward, scale={cfg_scale}, "
              f"null_label={num_classes}", flush=True)
    else:
        model_fn = model.forward
        print("CFG disabled: plain conditional sampling", flush=True)

    n = args.num_samples
    # Sharding: each worker generates the contiguous global-index slice [s0, s1).
    # Class labels come from the SAME global arange(n) mapping on every shard, so
    # pooling the shards reproduces the exact even-class coverage of a single run.
    # Noise is seeded per-shard (seed + s0) so shards are independent yet
    # reproducible; with the default s0=0 this is identical to the old single run.
    s0 = args.shard_start
    s1 = args.shard_end if args.shard_end is not None else n
    labels = (torch.arange(n) % config.misc.num_classes)[s0:s1]   # even class coverage
    g = torch.Generator(device=device).manual_seed(args.seed + s0)
    arr = np.empty((len(labels), config.training.image_size, config.training.image_size, 3), dtype=np.uint8)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(labels), args.batch):
            y = labels[i:i + args.batch].to(device)
            zs = torch.randn(len(y), *latent_size, device=device, generator=g)
            latents = ode(zs, model_fn, context=y, attn_mask=None)[-1]
            imgs = rae.decode(latents.float()).clamp(0, 1)
            arr[i:i + len(y)] = _to_uint8_nhwc(imgs)
            if (i // args.batch) % 10 == 0:
                print(f"  [shard {s0}:{s1}] generated {s0 + i + len(y)}/{s1} (global)", flush=True)

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
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--batch",       type=int, default=64)
    ap.add_argument("--steps",       type=int, default=50)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--raw",         action="store_true", help="Use raw weights instead of EMA")
    ap.add_argument("--grid",        default=None, help="Optional PNG path for a 64-sample grid")
    ap.add_argument("--out",         default=None, help="Optional JSON path for the result")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device",      default="cuda:0")
    # --- multi-GPU sharding ---
    # Workflow: launch one process per GPU with --shard-start/--shard-end + --gen-out
    # (each saves its uint8 [k,H,W,3] slice as .npy and skips FID), then one final
    # process with --pool "a.npy,b.npy,..." concatenates the shards, builds the shared
    # reference once, and computes a single FID. --num-samples must be the GLOBAL total
    # on every call so labels/reference stay aligned across shards.
    ap.add_argument("--shard-start", type=int, default=0, help="First global sample index (inclusive)")
    ap.add_argument("--shard-end",   type=int, default=None, help="Last global sample index (exclusive); default num_samples")
    ap.add_argument("--gen-out",     default=None, help="Generate this shard, save uint8 .npy, skip FID")
    ap.add_argument("--pool",        default=None, help="Comma-separated .npy shard files -> pool + FID")
    ap.add_argument("--cfg-scale",   type=float, default=None,
                    help="Standard CFG scale (cond/uncond double-forward). None/1.0 = plain conditional.")
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

    # --- pool mode: load shards, build reference once, compute one FID ---
    if args.pool:
        files = [f for f in args.pool.split(",") if f]
        print(f"Pooling {len(files)} shards: {files}", flush=True)
        gen_arr = np.concatenate([np.load(f) for f in files], axis=0)
        epoch = "?"
        print(f"Pooled generated set: {gen_arr.shape[0]} samples", flush=True)
        assert gen_arr.shape[0] == args.num_samples, \
            f"pooled {gen_arr.shape[0]} != --num-samples {args.num_samples} (shards incomplete?)"
        ref_arr = load_reference(args, config.training.image_size)
        print("Computing FID (torch-fidelity)...", flush=True)
        fid = calculate_rfid(gen_arr, ref_arr, bs=64, device="cuda")
        _g = (f"CFG cond/uncond, scale={args.cfg_scale}" if args.cfg_scale and args.cfg_scale != 1.0
              else "none (plain conditional Euler)")
        result = {"fid": fid, "ckpt": args.ckpt, "epoch": epoch,
                  "num_samples": args.num_samples, "steps": args.steps, "seed": args.seed,
                  "guidance": _g, "cfg_scale": args.cfg_scale,
                  "weights": "raw" if args.raw else "ema", "pooled_shards": files}
        print(f"FID = {fid:.3f}  ({args.num_samples} gen vs {args.num_samples} real, "
              f"ckpt={args.ckpt})", flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Result -> {args.out}", flush=True)
        return

    gen_arr, epoch = generate(args, config, device)

    # --- gen-only (shard) mode: save slice, skip reference + FID ---
    if args.gen_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.gen_out)), exist_ok=True)
        np.save(args.gen_out, gen_arr)
        print(f"Shard [{args.shard_start}:{args.shard_end}] -> {args.gen_out} "
              f"({gen_arr.shape[0]} samples)", flush=True)
        return

    ref_arr = load_reference(args, config.training.image_size)

    print("Computing FID (torch-fidelity)...", flush=True)
    fid = calculate_rfid(gen_arr, ref_arr, bs=64, device="cuda")

    _g = (f"CFG cond/uncond, scale={args.cfg_scale}" if args.cfg_scale and args.cfg_scale != 1.0
          else "none (plain conditional Euler)")
    result = {
        "fid": fid,
        "ckpt": args.ckpt,
        "epoch": epoch,
        "num_samples": args.num_samples,
        "steps": args.steps,
        "seed": args.seed,
        "guidance": _g,
        "cfg_scale": args.cfg_scale,
        "weights": "raw" if args.raw else "ema",
    }
    print(f"FID = {fid:.3f}  ({args.num_samples} gen vs {args.num_samples} real, "
          f"ckpt={args.ckpt}, epoch={epoch})", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
