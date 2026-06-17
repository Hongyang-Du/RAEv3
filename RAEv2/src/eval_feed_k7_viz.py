#!/usr/bin/env python3
"""Feed the SAME k=7 layers [11,13,15,17,19,21,23] to 3 decoders, reconstruct a few
val images, and SAVE the recon arrays + per-image PSNR to an npz (memory-safe: each
decoder is loaded, used, and freed in turn, so it fits beside a running job).
Panels: original | official k7 | official k23 (fed 7) | ours k23 no-SIGReg (fed 7).
The cute figure is drawn separately by plot_feed_k7_cute.py."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from omegaconf import OmegaConf

from utils.model_utils import instantiate_from_config, get_obj_from_str
from encoders.vision_encoder import create_encoder
from stage1.rae import _load_decoder

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
K7 = [11, 13, 15, 17, 19, 21, 23]


def psnr(rec, ref):
    mse = ((rec - ref) ** 2).flatten(1).mean(1)
    return (-10 * torch.log10(mse.clamp_min(1e-10))).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-viz", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--val-npz", default="data_eval/imagenet-256-val.npz")
    ap.add_argument("--images", default=None, help="comma-sep 256x256 png paths (overrides val-npz)")
    ap.add_argument("--layers", default="11,13,15,17,19,21,23", help="which DINOv3 layers to feed all 3 decoders")
    ap.add_argument("--out", default="output_full/feed_k7_viz.npz")
    args = ap.parse_args()
    K7 = [int(x) for x in args.layers.split(",")]
    print(f"feeding layers: {K7}", flush=True)
    dev = torch.device("cuda")
    mean, std = MEAN.to(dev), STD.to(dev)

    if args.images:
        from PIL import Image
        paths = [p.strip() for p in args.images.split(",") if p.strip()]
        imgs = torch.stack([torch.from_numpy(
            np.array(Image.open(p).convert("RGB").resize((256, 256)))) for p in paths])
        imgs = imgs.permute(0, 3, 1, 2).float().to(dev) / 255
        idxs = list(range(len(paths)))
    else:
        arr = np.load(args.val_npz, mmap_mode="r")
        arr = arr[arr.files[0]] if hasattr(arr, "files") else arr
        g = torch.Generator().manual_seed(args.seed)
        idxs = torch.randperm(len(arr), generator=g)[:args.n_viz].tolist()
        imgs = torch.stack([torch.from_numpy(arr[j].copy()) for j in idxs]).permute(0, 3, 1, 2).float().to(dev) / 255

    panels, psnrs = {"original": imgs.cpu().numpy()}, {}

    def run(tag, fn):
        recs = []
        for i in range(0, imgs.shape[0], args.batch):
            chunk = imgs[i:i + args.batch]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                recs.append(fn(chunk).clamp(0, 1).float())
        rec = torch.cat(recs, 0)
        panels[tag] = rec.cpu().numpy()
        psnrs[tag] = psnr(rec, imgs)
        print(f"{tag}: PSNR {psnrs[tag].mean():.2f} dB", flush=True)
        torch.cuda.empty_cache()

    # A) official k7 (native 7-layer decoder), fed the chosen layers
    rae = instantiate_from_config(OmegaConf.load("configs/stage1/sampling/dinov3l-k7-imagenet.yaml").stage_1).to(dev).eval()
    rae.encoder.layer_indices = list(K7)
    run("official_k7", lambda x: rae(x))
    del rae; torch.cuda.empty_cache()

    # B) official k23 fed only the 7 layers
    rae = instantiate_from_config(OmegaConf.load("configs/stage1/sampling/dinov3l-k23-imagenet.yaml").stage_1).to(dev).eval()
    rae.encoder.layer_indices = list(K7)
    run("official_k23_fed7", lambda x: rae(x))
    del rae; torch.cuda.empty_cache()

    # C) our k23 no-SIGReg (plain), fed the 7 layers via combine idx subset
    enc = create_encoder("dinov3mls-vit-l16[layers=" + ".".join(str(i) for i in range(1, 24)) + "]",
                         device=dev, resolution=256).eval()
    infer = OmegaConf.load("configs/stage1/decoder/infer-k23plain-fed-k7.yaml")
    ck = torch.load(infer.eval.ckpt, map_location="cpu", weights_only=False)
    layers = list(ck["layers"]); idx = [layers.index(L) for L in K7]
    combine = get_obj_from_str(infer.combine.target)(**OmegaConf.to_container(infer.combine.params, resolve=True)).to(dev).eval()
    combine.load_state_dict(ck["ema_combine"])
    dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16, num_patches=256, pretrained_path=None).to(dev).eval()
    dec.load_state_dict(ck["ema_dec"]); del ck

    def ours(x):
        toks = list(enc.model.get_intermediate_layers((x - mean) / std, n=layers, reshape=False,
                                                       return_class_token=False, norm=True))
        z = combine(toks, idx=idx)
        return dec.unpatchify(dec(z, drop_cls_token=False).logits) * std + mean
    run("ours_no_sigreg", ours)

    np.savez(args.out, idxs=np.array(idxs),
             **{f"img_{k}": v for k, v in panels.items()},
             **{f"psnr_{k}": v for k, v in psnrs.items()})
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
