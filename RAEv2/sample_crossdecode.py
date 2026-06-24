#!/usr/bin/env python3
"""Sample latents from the OFFICIAL k7 DiT, decode the SAME latents with two
decoders for comparison:
  (A) RAEv2-k7 decoder (native, the DiT's own stage-1)  -> RAE.decode (bare unpatchify)
  (B) ours OmniRAE cls-on decoder (RAEv2-trained)        -> unpatchify * std + mean

The DiT latent lives in the k7-normalized DINOv3-L space; both decoders un-normalize
with the SAME k7 stats. ours decoder sees raw k7 z == its feed-k7 condition (robust).

  python sample_crossdecode.py --classes 207 360 387 974 88 979 417 279 --per-class 4 \
      --steps 50 --out assets/viz_pca/crossdec_k7
"""
import argparse
import dataclasses
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

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
    ap.add_argument("--classes", type=int, nargs="+", default=[207, 360, 387, 974, 88, 979, 417, 279])
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ours-ckpt", default=OURS_CKPT)
    ap.add_argument("--out", default="assets/viz_pca/crossdec_k7")
    args = ap.parse_args()
    dev = "cuda"

    cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(CFG)))
    cfg.post_process(); cfg.prepare_model_params()

    rae = instantiate_from_config(cfg.stage_1).to(dev).eval()        # RAE-k7 (decoder + k7 stats)
    model = instantiate_from_config(cfg.stage_2).to(dev).eval()
    ck = torch.load(DIT_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(strip(ck["ema"])); print(f"DiT loaded ep{ck.get('epoch')}", flush=True); del ck

    # ours decoder (RAEv2-trained -> needs *std+mean)
    dec_ours = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                             num_patches=256, pretrained_path=None).to(dev).eval()
    ock = torch.load(args.ours_ckpt, map_location="cpu", weights_only=False)
    dec_ours.load_state_dict(ock["ema_dec"]); print("ours decoder loaded", flush=True); del ock
    mean, std = MEAN.to(dev), STD.to(dev)

    latent_size = tuple(cfg.misc.latent_size)
    tds = math.sqrt((cfg.misc.time_dist_shift_dim or math.prod(latent_size)) / cfg.misc.time_dist_shift_base)
    transport = create_transport(config=cfg.transport, time_dist_shift=tds)
    sampler = create_sampler(transport, guidance_config=cfg.guidance)
    ode = sampler.sample_ode(**{**dataclasses.asdict(cfg.sampler), "num_steps": args.steps})

    y = torch.tensor([c for c in args.classes for _ in range(args.per_class)], device=dev)
    n = y.shape[0]
    g = torch.Generator(device=dev).manual_seed(args.seed)
    zs = torch.randn(n, *latent_size, device=dev, generator=g)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        latents = ode(zs, model.forward, context=y, attn_mask=None)[-1].float()
        # (A) native k7 decoder via RAE.decode (un-norm k7 + bare unpatchify)
        img_k7 = rae.decode(latents).clamp(0, 1)
        # (B) ours: same k7 un-norm, then ours decoder, then *std+mean
        lm = rae.latent_mean.to(dev) if rae.latent_mean is not None else 0
        lv = rae.latent_var.to(dev) if rae.latent_var is not None else 1
        z_raw = latents * torch.sqrt(lv + rae.eps) + lm                 # [n,1024,16,16]
        z_raw = z_raw.view(n, 1024, 256).transpose(1, 2)                # [n,256,1024]
        out = dec_ours(z_raw, drop_cls_token=False).logits
        img_ours = (dec_ours.unpatchify(out) * std + mean).clamp(0, 1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    nrow = args.per_class
    save_image(img_k7.cpu(), f"{args.out}_k7native.png", nrow=nrow)
    save_image(img_ours.cpu(), f"{args.out}_ours.png", nrow=nrow)
    # interleaved side-by-side (k7 | ours per sample column-pair)
    inter = torch.stack([img_k7, img_ours], dim=1).reshape(2 * n, *img_k7.shape[1:])
    save_image(inter.cpu(), f"{args.out}_pair.png", nrow=2 * nrow)
    print(f"saved -> {args.out}_k7native.png / _ours.png / _pair.png  (n={n})", flush=True)


if __name__ == "__main__":
    main()
