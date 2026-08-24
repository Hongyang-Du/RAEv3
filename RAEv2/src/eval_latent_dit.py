#!/usr/bin/env python3
"""Decoupled stage-2 evaluation: dFID + latent-space distribution metrics.

Motivation: standard gFID mixes the stage-1 reconstruction bottleneck with the
stage-2 fit. This script separates them.

Pipeline (single GPU):
  1. Sample N latents from the stage-2 DiT (same Euler ODE path and guidance
     handling as eval_fid_dit.py) and decode with the stage-1 decoder.
  2. Encode N real images with the stage-1 encoder (real latents) and decode
     them back (reconstructions). Cacheable via --real-cache for sweeps.
  3. Pixel space (torch-fidelity):
       rfid = FID(recon, real)   stage-1 reconstruction bottleneck
       gfid = FID(gen,   real)   standard generation FID
       dfid = FID(gen,   recon)  decoder-decoupled generation FID
     Triangle bound (Gaussian approx): sqrt(gfid) <= sqrt(rfid) + sqrt(dfid).
  4. Latent space (eval.latent_metrics, real vs sampled latents, normalized
     DiT space): per-token FD (aggregate), per-position FD heatmap, pooled FD,
     random-projection FD + RBF-MMD (the RP pair sees cross-token structure).

Reading the numbers: if per-token/pooled/rp FD and MMD are all low but dfid is
high, stage-2 has fit the latent distribution and the remaining gap is decoder
manifold roughness — not diffusion underfitting.

Usage (inside rae container, one free GPU):
    python src/eval_latent_dit.py \
        --config configs/stage2/training/imagenet-dinov3l-encoder-cls-k23.yaml \
        --ckpt   ckpts_full/stage2/dit-l11/checkpoints/ep-0000009.pt \
        --data   /datasets/imagenet-256-full \
        --num-samples 10000 \
        --real-cache ckpts_full/eval_cache/real10k_seed42.npz \
        --out    ckpts_full/stage2/dit-l11/latent_eval.json
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

from configs.stage2 import Stage2Config
from eval.fid import calculate_fid_isc, calculate_rfid
from eval.latent_metrics import compute_all
from eval_fid_dit import _build_guidance, _strip_prefixes, _to_uint8_nhwc, load_reference
from stage2.transport import create_sampler, create_transport
from utils.model_utils import instantiate_from_config


def generate_with_latents(args, config, rae, device):
    """eval_fid_dit.generate, but also keeps the sampled latents (fp16, the
    normalized space the DiT runs in — i.e. what rae.decode consumes)."""
    from utils.guidance_utils import get_model_forward_fn
    model = instantiate_from_config(config.stage_2).to(device).eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = _strip_prefixes(ck["model" if args.raw else "ema"])
    model.load_state_dict(sd)
    epoch = ck.get("epoch", "?")
    del ck, sd
    print(f"Loaded {'raw' if args.raw else 'EMA'} DiT from {args.ckpt} (epoch {epoch})", flush=True)

    guid = _build_guidance(args)
    model_fn, sample_kwargs = get_model_forward_fn(model, guid)
    use_guidance = guid.any_guidance_active

    latent_size = tuple(config.misc.latent_size)
    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size)) / config.misc.time_dist_shift_base)
    transport = create_transport(config=config.transport, time_dist_shift=time_dist_shift)
    sampler = create_sampler(transport, guidance_config=guid)
    ode = sampler.sample_ode(**{**dataclasses.asdict(config.sampler), "num_steps": args.steps})

    n = args.num_samples
    null_label = config.misc.num_classes
    labels = torch.arange(n) % config.misc.num_classes
    g = torch.Generator(device=device).manual_seed(args.seed)
    imgs_arr = np.empty((n, config.training.image_size, config.training.image_size, 3), dtype=np.uint8)
    lat_arr = np.empty((n, *latent_size), dtype=np.float16)

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
            latents = latents.float()
            lat_arr[i:i + bs] = latents.half().cpu().numpy()
            imgs_arr[i:i + bs] = _to_uint8_nhwc(rae.decode(latents).clamp(0, 1))
            if (i // args.batch) % 10 == 0:
                print(f"  generated {i + bs}/{n}", flush=True)

    del model
    torch.cuda.empty_cache()
    return lat_arr, imgs_arr, epoch


def build_real_side(args, config, rae, device):
    """Real images -> (real imgs uint8, real latents fp16, recon imgs uint8).
    Encode/decode run under the same bf16 autocast as generation so the
    decoder mapping is identical on both sides of dfid."""
    stage1_repr = str(config.stage_1)
    if args.real_cache and os.path.exists(args.real_cache):
        z = np.load(args.real_cache)
        if (int(z["num_samples"]) == args.num_samples and int(z["seed"]) == args.seed
                and str(z["stage1_repr"]) == stage1_repr):
            print(f"Loaded real cache {args.real_cache} "
                  f"(n={z['real_images'].shape[0]})", flush=True)
            return z["real_images"], z["real_latents"], z["recon_images"]
        print(f"WARN real cache mismatch (n={z['num_samples']}, seed={z['seed']}, "
              f"stage1 changed={str(z['stage1_repr']) != stage1_repr}), rebuilding", flush=True)

    if args.ref_npz:
        _z = np.load(args.ref_npz)
        real = _z["arr_0"] if "arr_0" in _z else _z[list(_z.keys())[0]]
        real = real[:args.num_samples]
        print(f"Loaded reference {real.shape} from {args.ref_npz}", flush=True)
    else:
        real = load_reference(args, config.training.image_size)

    n = real.shape[0]
    latent_size = tuple(config.misc.latent_size)
    lat_arr = np.empty((n, *latent_size), dtype=np.float16)
    recon_arr = np.empty_like(real)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, n, args.batch):
            x = torch.from_numpy(real[i:i + args.batch]).permute(0, 3, 1, 2).float().div(255).to(device)
            z_lat = rae.encode(x).float()
            lat_arr[i:i + x.shape[0]] = z_lat.half().cpu().numpy()
            recon_arr[i:i + x.shape[0]] = _to_uint8_nhwc(rae.decode(z_lat).clamp(0, 1))
            if (i // args.batch) % 20 == 0:
                print(f"  encoded/reconstructed {i + x.shape[0]}/{n}", flush=True)

    if args.real_cache:
        os.makedirs(os.path.dirname(os.path.abspath(args.real_cache)), exist_ok=True)
        np.savez(args.real_cache, real_images=real, real_latents=lat_arr,
                 recon_images=recon_arr, num_samples=args.num_samples, seed=args.seed,
                 stage1_repr=stage1_repr)
        print(f"Real cache -> {args.real_cache}", flush=True)
    return real, lat_arr, recon_arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      required=True, help="Stage-2 training YAML")
    ap.add_argument("--ckpt",        required=True, help="Stage-2 checkpoint (ep-*.pt)")
    ap.add_argument("--data",        default="/datasets/imagenet-256-full")
    ap.add_argument("--ref-npz",     default=None, help="uint8 NHWC npz as the real set (else sampled train images)")
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--batch",       type=int, default=64)
    ap.add_argument("--steps",       type=int, default=50)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--raw",         action="store_true", help="Use raw weights instead of EMA")
    ap.add_argument("--out",         default=None, help="JSON path for results")
    ap.add_argument("--real-cache",  default=None, help="npz cache for real imgs/latents/recons (reused across ckpts)")
    ap.add_argument("--save-gen",    default=None, help="Optional npz path to dump gen latents+images")
    ap.add_argument("--proj-dim",    type=int, default=2048)
    ap.add_argument("--mmd-max",     type=int, default=10000)
    ap.add_argument("--skip-pixel",  action="store_true", help="Latent metrics only (no Inception passes)")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device",      default="cuda:0")
    # Guidance (defaults => plain conditional, identical to eval_fid_dit)
    ap.add_argument("--ig-scale",   type=float, default=1.0)
    ap.add_argument("--ig-tmin",    type=float, default=0.0)
    ap.add_argument("--ig-tmax",    type=float, default=1.0)
    ap.add_argument("--uncond-ig-scale", type=float, default=None)
    ap.add_argument("--cfg-scale",  type=float, default=1.0)
    ap.add_argument("--cfg-tmin",   type=float, default=0.0)
    ap.add_argument("--cfg-tmax",   type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device(args.device)
    config: Stage2Config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config)))
    config.post_process()
    config.prepare_model_params()

    rae = instantiate_from_config(config.stage_1).to(device).eval()

    gen_lat, gen_imgs, epoch = generate_with_latents(args, config, rae, device)
    if args.save_gen:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_gen)), exist_ok=True)
        np.savez(args.save_gen, gen_latents=gen_lat, gen_images=gen_imgs)
        print(f"Gen dump -> {args.save_gen}", flush=True)

    real_imgs, real_lat, recon_imgs = build_real_side(args, config, rae, device)
    del rae
    torch.cuda.empty_cache()

    result = {
        "ckpt": args.ckpt, "epoch": epoch, "num_samples": args.num_samples,
        "steps": args.steps, "seed": args.seed,
        "weights": "raw" if args.raw else "ema",
        "cfg_scale": args.cfg_scale, "ig_scale": args.ig_scale,
    }

    if not args.skip_pixel:
        print("Pixel FIDs (torch-fidelity)...", flush=True)
        gfid, isc = calculate_fid_isc(gen_imgs, real_imgs, bs=64, device="cuda")
        rfid = calculate_rfid(recon_imgs, real_imgs, bs=64, device="cuda")
        dfid = calculate_rfid(gen_imgs, recon_imgs, bs=64, device="cuda")
        result.update({
            "gfid": gfid, "rfid": rfid, "dfid": dfid, "is": isc,
            "sqrt_gfid": math.sqrt(gfid),
            "sqrt_rfid_plus_sqrt_dfid": math.sqrt(rfid) + math.sqrt(dfid),
        })
        print(f"gFID={gfid:.3f}  rFID={rfid:.3f}  dFID={dfid:.3f}  IS={isc:.2f}  "
              f"(triangle: {math.sqrt(gfid):.3f} <= {math.sqrt(rfid) + math.sqrt(dfid):.3f})", flush=True)

    print("Latent metrics...", flush=True)
    lat_metrics, heatmap = compute_all(
        real_lat, gen_lat, device=args.device,
        proj_dim=args.proj_dim, mmd_max_samples=args.mmd_max)
    result["latent"] = lat_metrics
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in lat_metrics.items()
                           if isinstance(v, float)), flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        base = os.path.splitext(args.out)[0]
        np.save(base + "_pos_fd.npy", heatmap)
        print(f"Result -> {args.out}  (heatmap -> {base}_pos_fd.npy)", flush=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(heatmap, cmap="viridis")
            ax.set_title("per-position latent FD")
            fig.colorbar(im, ax=ax)
            fig.savefig(base + "_pos_fd.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Heatmap PNG -> {base}_pos_fd.png", flush=True)
        except Exception as e:
            print(f"(heatmap PNG skipped: {e})", flush=True)


if __name__ == "__main__":
    main()
