#!/usr/bin/env python3
"""Shardable: sample class-balanced latents from the official k7 DiT, decode each
with (A) native RAEv2-k7 decoder and (B) ours OmniRAE cls-on decoder, save uint8
arrays + per-image agreement (PSNR/SSIM/LPIPS native-vs-ours) for the shard.

  CUDA_VISIBLE_DEVICES=i python gen_crossdecode.py --num 10000 --nshards 8 --shard i \
      --batch 16 --steps 50 --out-dir output_full/crossdec_k7
"""
import argparse
import dataclasses
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch
import lpips as lpips_lib
from omegaconf import OmegaConf
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

from configs.stage2 import Stage2Config
from stage2.transport import create_sampler, create_transport
from utils.model_utils import instantiate_from_config
from stage1.rae import _load_decoder

CFG = "configs/stage2/training/imagenet-dinov3l-k7.yaml"
DIT_CKPT = "pretrained_models/stage2/imagenet/dinov3l-k7/checkpoint.pt"
OURS_CKPT = "output_full/decoder_random_drop_layer_mls_plain_cls_k23_16ep/ckpt_ep016.pt"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def strip(sd):
    out = {}
    for k, v in sd.items():
        for p in ("module.", "_orig_mod."):
            while k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=10000)
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--ours-ckpt", default=OURS_CKPT)
    ap.add_argument("--out-dir", default="output_full/crossdec_k7")
    args = ap.parse_args()
    dev = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)

    per_class = args.num // 1000
    glob_idx = [(c, j) for c in range(1000) for j in range(per_class)]   # class-balanced
    mine = [(gi, cj) for gi, cj in enumerate(glob_idx) if gi % args.nshards == args.shard]
    print(f"[shard {args.shard}/{args.nshards}] {len(mine)} samples", flush=True)

    cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(CFG)))
    cfg.post_process(); cfg.prepare_model_params()
    rae = instantiate_from_config(cfg.stage_1).to(dev).eval()
    model = instantiate_from_config(cfg.stage_2).to(dev).eval()
    ck = torch.load(DIT_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(strip(ck["ema"])); del ck
    dec_ours = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                             num_patches=256, pretrained_path=None).to(dev).eval()
    ock = torch.load(args.ours_ckpt, map_location="cpu", weights_only=False)
    dec_ours.load_state_dict(ock["ema_dec"]); del ock
    lp = lpips_lib.LPIPS(net="vgg").to(dev).eval()
    mean, std = MEAN.to(dev), STD.to(dev)
    lm = rae.latent_mean.to(dev) if rae.latent_mean is not None else 0
    lv = rae.latent_var.to(dev) if rae.latent_var is not None else 1

    latent_size = tuple(cfg.misc.latent_size)
    tds = math.sqrt((cfg.misc.time_dist_shift_dim or math.prod(latent_size)) / cfg.misc.time_dist_shift_base)
    transport = create_transport(config=cfg.transport, time_dist_shift=tds)
    sampler = create_sampler(transport, guidance_config=cfg.guidance)
    ode = sampler.sample_ode(**{**dataclasses.asdict(cfg.sampler), "num_steps": args.steps})

    nat_all, our_all = [], []
    ps = ss = lpv = 0.0
    done = 0
    for i in range(0, len(mine), args.batch):
        chunk = mine[i:i + args.batch]
        y = torch.tensor([cj[0] for _, cj in chunk], device=dev)
        zs = torch.stack([torch.randn(*latent_size, generator=torch.Generator(device=dev).manual_seed(1000 + gi),
                                       device=dev) for gi, _ in chunk])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lat = ode(zs, model.forward, context=y, attn_mask=None)[-1].float()
            img_nat = rae.decode(lat).clamp(0, 1)
            zr = (lat * torch.sqrt(lv + rae.eps) + lm).view(lat.shape[0], 1024, 256).transpose(1, 2)
            out = dec_ours(zr, drop_cls_token=False).logits
            img_our = (dec_ours.unpatchify(out) * std + mean).clamp(0, 1)
        # agreement (ours vs native as reference)
        mse = ((img_our - img_nat) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
        ps += (-10 * torch.log10(mse)).sum().item()
        ss += ssim_fn(img_our, img_nat, data_range=1.0).item() * img_nat.shape[0]
        with torch.no_grad():
            lpv += lp(img_our * 2 - 1, img_nat * 2 - 1).sum().item()
        nat_all.append((img_nat * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy())
        our_all.append((img_our * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy())
        done += len(chunk)
        if done % (args.batch * 10) == 0:
            print(f"  [{args.shard}] {done}/{len(mine)}", flush=True)

    np.save(f"{args.out_dir}/native_{args.shard}.npy", np.concatenate(nat_all))
    np.save(f"{args.out_dir}/ours_{args.shard}.npy", np.concatenate(our_all))
    np.savez(f"{args.out_dir}/agree_{args.shard}.npz", psnr=ps, ssim=ss, lpips=lpv, n=done)
    print(f"[shard {args.shard}] saved {done}  PSNR={ps/done:.2f} SSIM={ss/done:.3f} LPIPS={lpv/done:.3f}", flush=True)


if __name__ == "__main__":
    main()
