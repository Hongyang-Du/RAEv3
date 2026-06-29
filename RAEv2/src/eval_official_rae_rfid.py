"""Reconstruction PSNR/SSIM/rFID for OFFICIAL RAEv2 stage1.RAE decoders
(decoder.pt + stats.pt), under a chosen feed condition (encoder layer subset).

Uses the official stage1.RAE class directly (so preprocessing, cls surrogate, and
stats normalization exactly match the released models), and the SAME val NPZ +
calculate_rfid as eval_recon_subset_rfid.py so numbers are directly comparable.

The feed condition is set by the encoder's layer list (encoder_name layers=...),
which is how RAEv2 implements feed k=7 / k=23 / l11.

Usage:
  python src/eval_official_rae_rfid.py \
    --decoder <decoder.pt> --stats <stats.pt> \
    --feed-layers 1,2,...,23 --num-images 50000 --tag k23_feedk23 --out out.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

from stage1.rae import RAE
from eval.fid import calculate_rfid


def per_image_psnr(rec, ref):
    mse = ((rec.clamp(0, 1) - ref) ** 2).mean(dim=(1, 2, 3))
    return -10 * torch.log10(mse.clamp_min(1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--stats", required=False, default=None)
    ap.add_argument("--feed-layers", required=True, help="comma-sep encoder layers, e.g. 1,2,...,23")
    ap.add_argument("--val-npz", default="data_eval/imagenet-256-val.npz")
    ap.add_argument("--num-images", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"

    feed_layers = [int(x) for x in args.feed_layers.split(",")]
    layers_str = ".".join(map(str, feed_layers))
    print(f"[{args.tag}] feed_layers={feed_layers}", flush=True)

    # Official RAE: encoder layer list = feed condition; decoder + stats from release.
    rae = RAE(
        encoder_name=f"dinov3mls-vit-l16[layers={layers_str}]",
        resolution=256,
        decoder_config_path="configs/decoder/ViTXL",
        pretrained_decoder_path=args.decoder,
        normalization_stat_path=args.stats,
        noise_tau=0.0,
    ).to(device).eval()

    arr = np.load(args.val_npz, mmap_mode="r")
    arr = arr[arr.files[0]] if hasattr(arr, "files") else arr
    g = torch.Generator().manual_seed(args.seed)
    idxs = torch.randperm(len(arr), generator=g)[:args.num_images].tolist()

    psnrs, ssims, n_done = [], [], 0
    recon_u8, ref_u8 = [], []
    with torch.no_grad():
        for i in range(0, len(idxs), args.batch):
            bi = idxs[i:i + args.batch]
            imgs = torch.stack([torch.from_numpy(arr[j].copy()) for j in bi])
            imgs = imgs.permute(0, 3, 1, 2).float().to(device) / 255   # [0,1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rec = rae(imgs)                                        # official encode->decode
            rec = rec.clamp(0, 1).float()
            psnrs.append(per_image_psnr(rec, imgs))
            ssims.append(ssim_fn(rec, imgs, data_range=1.0).item() * rec.shape[0])
            recon_u8.append(rec.mul(255).round().permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy())
            ref_u8.append(imgs.mul(255).round().permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy())
            n_done += imgs.shape[0]
            if n_done % 3200 == 0:
                print(f"  {n_done}/{len(idxs)}", flush=True)

    psnr = torch.cat(psnrs)
    ssim = sum(ssims) / n_done
    recon_arr = np.concatenate(recon_u8, 0)
    ref_arr = np.concatenate(ref_u8, 0)
    rfid = calculate_rfid(recon_arr, ref_arr, bs=128, device=device)

    print(f"\n[{args.tag}] N={n_done}  feed_layers={feed_layers}")
    print(f"  PSNR = {psnr.mean():.3f} dB   SSIM = {ssim:.4f}   rFID = {rfid:.3f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"decoder": args.decoder, "tag": args.tag, "feed_layers": feed_layers,
                   "num_images": n_done, "psnr": psnr.mean().item(), "ssim": ssim,
                   "rfid": float(rfid)}, f, indent=2)
    print(f"json -> {args.out}")


if __name__ == "__main__":
    main()
