#!/usr/bin/env python3
"""Reconstruction PSNR + SSIM for an OFFICIAL raev2 RAE (stage1.RAE, loaded from a
sampling config), on the SAME fixed val-npz subset used by eval_recon_subset.py
(seed 0) so the official k7/k23 decoders are directly comparable to ours.

Usage:
  python src/eval_official_recon.py --config configs/stage1/sampling/dinov3l-k7-imagenet.yaml --tag official_k7
  python src/eval_official_recon.py --config configs/stage1/sampling/dinov3l-k23-imagenet.yaml --tag official_k23
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from omegaconf import OmegaConf
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

from utils.model_utils import instantiate_from_config


def per_image_psnr(rec, ref):
    mse = ((rec.clamp(0, 1) - ref) ** 2).mean(dim=(1, 2, 3))
    return -10 * torch.log10(mse.clamp_min(1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="official sampling YAML (has stage_1 block)")
    ap.add_argument("--val-npz", default="data_eval/imagenet-256-val.npz")
    ap.add_argument("--num-images", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="official")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device("cuda")

    cfg = OmegaConf.load(args.config)
    rae = instantiate_from_config(cfg.stage_1).to(device).eval()
    for p in rae.parameters():
        p.requires_grad_(False)
    print(f"[{args.tag}] encoder={cfg.stage_1.params.encoder_name}", flush=True)

    arr = np.load(args.val_npz, mmap_mode="r")
    arr = arr[arr.files[0]] if hasattr(arr, "files") else arr      # [N,256,256,3] uint8
    g = torch.Generator().manual_seed(args.seed)
    idxs = torch.randperm(len(arr), generator=g)[:args.num_images].tolist()

    psnrs, ssims, n_done = [], [], 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(idxs), args.batch):
            bi = idxs[i:i + args.batch]
            imgs = torch.stack([torch.from_numpy(arr[j].copy()) for j in bi])
            imgs = imgs.permute(0, 3, 1, 2).float().to(device) / 255   # [B,3,256,256] in [0,1]
            rec = rae(imgs).float().clamp(0, 1)
            psnrs.append(per_image_psnr(rec, imgs))
            ssims.append(ssim_fn(rec, imgs, data_range=1.0).item() * rec.shape[0])
            n_done += imgs.shape[0]
            if n_done % 320 == 0:
                print(f"  {n_done}/{len(idxs)}", flush=True)

    psnr = torch.cat(psnrs)
    ssim = sum(ssims) / n_done
    print(f"\n[{args.tag}] N={n_done}")
    print(f"  PSNR = {psnr.mean():.3f} +/- {psnr.std():.3f} dB  (median {psnr.median():.3f})")
    print(f"  SSIM = {ssim:.4f}")

    out = args.out or f"output_full/official_recon_{args.tag}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": args.config, "tag": args.tag, "num_images": n_done,
                   "psnr_mean": psnr.mean().item(), "psnr_std": psnr.std().item(),
                   "ssim": ssim}, f, indent=2)
    print(f"json -> {out}")


if __name__ == "__main__":
    main()
