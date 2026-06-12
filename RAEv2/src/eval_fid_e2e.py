#!/usr/bin/env python3
"""Offline generation FID for an END-TO-END checkpoint (train_e2e_sigreg_dit.py).

Same protocol as eval_fid_dit.py (N class-balanced samples, fixed seed, 50-step
Euler with the SHIFTED time grid the official stage-2 sampler uses, vs N real
train images) so e2e FID is directly comparable with the stage-2 sweep numbers.
Differences vs eval_fid_dit.py:
  - loads ema_dit + live decoder from the single e2e ckpt (no stage-2 config YAML)
  - latent space is the RAW live z (SIGReg'd ~N(0,1)); no stats normalization

Usage (inside rae container, one free GPU):
    python src/eval_fid_e2e.py \
        --ckpt output_full/train_e2e_nodrop/ckpt_latest.pt \
        --data /datasets/imagenet-256-full \
        --num-samples 5000 --out output_full/train_e2e_nodrop/fid.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torchvision.utils import save_image

from configs.stage2 import ConditioningArchConfig
from eval.fid import calculate_rfid
from eval_fid_dit import _to_uint8_nhwc, load_reference
from stage1.rae import _load_decoder
from stage2.models.DDT import DiTwDDTHeadIG


def spatial_to_tokens(z):                                   # [B,C,H,W] -> [B,N,C]
    b, c, h, w = z.shape
    return z.view(b, c, h * w).transpose(1, 2)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def euler_sample_shifted(dit, noise, labels, num_steps=50, t_eps=0.05, shift=8.0):
    """Euler ODE with the official stage-2 shifted time grid (denser near t=0)."""
    x = noise.clone()
    u = torch.linspace(1.0, 0.0, num_steps + 1, device=x.device)
    ts = shift * u / (1 + (shift - 1) * u)
    for i in range(num_steps):
        t, dt = ts[i], ts[i] - ts[i + 1]
        pred = dit(x, t.expand(x.shape[0]), context=labels, attn_mask=None)
        if isinstance(pred, tuple):
            pred = pred[0]
        v = (x - pred) / max(t.item(), t_eps)
        x = x - dt * v
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",        required=True, help="e2e ckpt (ckpt_latest.pt / ckpt_epXXX.pt)")
    ap.add_argument("--data",        default="/datasets/imagenet-256-full")
    ap.add_argument("--num-samples", type=int, default=5000)
    ap.add_argument("--batch",       type=int, default=256)
    ap.add_argument("--steps",       type=int, default=50)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--t-shift",     type=float, default=8.0)
    ap.add_argument("--latent-dim",  type=int, default=1024)
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--image-size",  type=int, default=256)
    ap.add_argument("--raw",         action="store_true", help="raw weights instead of EMA")
    ap.add_argument("--grid",        default=None, help="optional PNG for a 64-sample grid")
    ap.add_argument("--out",         default=None, help="optional JSON path for the result")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device",      default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    epoch = ck.get("epoch", "?")

    dit = DiTwDDTHeadIG(
        input_size=16, patch_size=[1, 1], in_channels=args.latent_dim,
        hidden_size=[1440, 2048], depth=[28, 2], num_heads=[20, 16], mlp_ratio=4.0,
        base_model_depth=8, num_classes=args.num_classes,
        condition_type="label", context_dim=None,
        cond_arch=ConditioningArchConfig(num_t_tokens=4, num_c_tokens=8),
    ).to(device).eval()
    dit.load_state_dict(ck["dit" if args.raw else "ema_dit"])

    decoder = _load_decoder("configs/decoder/ViTXL", hidden_size=args.latent_dim,
                            patch_size=16, num_patches=256, pretrained_path=None).to(device).eval()
    decoder.load_state_dict(ck["decoder"])     # decoder is live-only (no EMA kept)
    for p in list(dit.parameters()) + list(decoder.parameters()):
        p.requires_grad_(False)
    print(f"Loaded {'raw' if args.raw else 'EMA'} dit + live decoder from {args.ckpt} (epoch {epoch})", flush=True)
    del ck

    img_mean, img_std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    n = args.num_samples
    labels = torch.arange(n) % args.num_classes
    g = torch.Generator(device=device).manual_seed(args.seed)
    arr = np.empty((n, args.image_size, args.image_size, 3), dtype=np.uint8)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, n, args.batch):
            y = labels[i:i + args.batch].to(device)
            zs = torch.randn(len(y), args.latent_dim, 16, 16, device=device, generator=g)
            z_gen = euler_sample_shifted(dit, zs, y, num_steps=args.steps, shift=args.t_shift)
            out = decoder(spatial_to_tokens(z_gen.float()), drop_cls_token=False).logits
            imgs = (decoder.unpatchify(out) * img_std + img_mean).clamp(0, 1)
            arr[i:i + len(y)] = _to_uint8_nhwc(imgs)
            if (i // args.batch) % 5 == 0:
                print(f"  generated {i + len(y)}/{n}", flush=True)

    if args.grid:
        os.makedirs(os.path.dirname(os.path.abspath(args.grid)), exist_ok=True)
        save_image(torch.from_numpy(arr[:64]).permute(0, 3, 1, 2).float() / 255, args.grid, nrow=8)
        print(f"Sample grid -> {args.grid}", flush=True)

    del dit, decoder
    torch.cuda.empty_cache()
    ref_arr = load_reference(args, args.image_size)

    print("Computing FID (torch-fidelity)...", flush=True)
    fid = calculate_rfid(arr, ref_arr, bs=64, device="cuda")
    result = {"fid": fid, "ckpt": args.ckpt, "epoch": epoch,
              "num_samples": n, "steps": args.steps, "seed": args.seed,
              "guidance": "none (plain conditional Euler, shifted grid)",
              "weights": "raw" if args.raw else "ema"}
    print(f"FID = {fid:.3f}  ({n} gen vs {n} real, ckpt={args.ckpt}, epoch={epoch})", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
