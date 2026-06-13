#!/usr/bin/env python3
"""Reconstruction PSNR + SSIM for an MLS-decoder stage-1 ckpt on the ImageNet
val set (full-mean combine, eval mode -> no layer dropout).

Loads ema_proj/ema_dec from a dropmean / dropmean_bn / nogate ckpt, runs the
frozen DINOv3 encoder, projects, decodes, and reports per-image PSNR + SSIM
averaged over N val images (data_eval/imagenet-256-val.npz, seed-fixed subset).

Usage (1 GPU):
    python src/eval_recon_psnr_ssim.py \
        --ckpt output_full/train_decoder_mls_dropmean_bn_all24/ckpt_ep004.pt \
        --variant dropmean_bn --num-images 5000
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def per_image_psnr(rec, ref):
    mse = ((rec.clamp(0, 1) - ref) ** 2).mean(dim=(1, 2, 3))
    return -10 * torch.log10(mse.clamp_min(1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="dropmean_bn",
                    choices=["dropmean_bn", "dropmean_ln", "nogate"])
    ap.add_argument("--val-npz", default="data_eval/imagenet-256-val.npz")
    ap.add_argument("--num-images", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device("cuda")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    layers = ck["layers"]
    print(f"{args.variant}: ckpt epoch {ck.get('epoch')}, layers={layers}", flush=True)

    from encoders.vision_encoder import create_encoder
    enc = create_encoder("dinov3mls-vit-l16[layers=" + ".".join(map(str, layers)) + "]",
                         device=device, resolution=256).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    from stage1.rae import _load_decoder
    dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                        num_patches=256, pretrained_path=None).to(device).eval()
    dec.load_state_dict(ck["ema_dec"])

    if args.variant == "dropmean_bn":
        from train_decoder_mls_dropmean_bn_sigreg import MLSProjector
        proj = MLSProjector(dim=enc.hidden_size, out_dim=1024)
    elif args.variant == "dropmean_ln":
        from train_decoder_mls_dropmean_sigreg import MLSProjector
        proj = MLSProjector(dim=enc.hidden_size, out_dim=1024)
    else:
        from train_decoder_mls_nogate_sigreg import MLSProjector
        proj = MLSProjector(dim=enc.hidden_size, out_dim=1024)
    proj.load_state_dict(ck["ema_proj"])
    proj.to(device).eval()
    for p in proj.parameters():
        p.requires_grad_(False)
    del ck

    arr = np.load(args.val_npz, mmap_mode="r")
    arr = arr[arr.files[0]] if hasattr(arr, "files") else arr   # [N,256,256,3] uint8
    g = torch.Generator().manual_seed(args.seed)
    idxs = torch.randperm(len(arr), generator=g)[:args.num_images].tolist()

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnrs, n_done = [], 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(idxs), args.batch):
            batch_idx = idxs[i:i + args.batch]
            imgs = torch.stack([torch.from_numpy(arr[j].copy()) for j in batch_idx])
            imgs = imgs.permute(0, 3, 1, 2).float().to(device) / 255      # [B,3,256,256]
            toks = list(enc.model.get_intermediate_layers(
                (imgs - mean) / std, n=layers, reshape=False,
                return_class_token=False, norm=True))
            z = proj(toks)                                                # full mean (eval)
            out = dec(z, drop_cls_token=False).logits
            rec = (dec.unpatchify(out) * std + mean).clamp(0, 1)
            psnrs.append(per_image_psnr(rec.float(), imgs))
            ssim_metric.update(rec.float(), imgs)
            n_done += imgs.shape[0]
            if n_done % 320 == 0:
                print(f"  {n_done}/{len(idxs)}", flush=True)

    psnr = torch.cat(psnrs)
    ssim = ssim_metric.compute().item()
    print(f"\n[{args.variant}] ckpt={os.path.basename(args.ckpt)}  N={n_done}")
    print(f"  PSNR = {psnr.mean():.3f} ± {psnr.std():.3f} dB  (median {psnr.median():.3f})")
    print(f"  SSIM = {ssim:.4f}")

    out = args.out or args.ckpt.replace(".pt", "_recon_val.json")
    with open(out, "w") as f:
        json.dump({"ckpt": args.ckpt, "variant": args.variant, "num_images": n_done,
                   "psnr_mean": psnr.mean().item(), "psnr_std": psnr.std().item(),
                   "ssim": ssim}, f, indent=2)
    print(f"json -> {out}")


if __name__ == "__main__":
    main()
