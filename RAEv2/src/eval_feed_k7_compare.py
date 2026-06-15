#!/usr/bin/env python3
"""Feed the SAME k=7 DINOv3 layers [11,13,15,17,19,21,23] to three decoders and
compare reconstruction PSNR/SSIM (+ a visualization):

  A) official raev2 k7 decoder   -- native (it was trained on exactly these 7 layers)
  B) official raev2 k23 decoder  -- fed only 7 layers (trained on 23 -> out-of-budget)
  C) our k23 drop-mean + SIGReg  -- fed only 7 layers via the combine idx subset

A & B reuse the official RAE pipeline (SUM combine + per-model normalization + decoder);
for B we just override encoder.layer_indices to the 7 layers. C uses our encoder +
MLSCombine(idx) + decoder. The per-layer DINOv3 tokens are identical across all three
(same backbone, same ImageNet norm); only the combine/decoder differ.

Single GPU. Writes a metrics table + output_full/feed_k7_compare.png (rows = sample
images, cols = original | official-k7 | official-k23(fed7) | ours-k23(fed7)).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

from utils.model_utils import instantiate_from_config, get_obj_from_str
from encoders.vision_encoder import create_encoder
from stage1.rae import _load_decoder

K7_LAYERS = [11, 13, 15, 17, 19, 21, 23]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def per_image_psnr(rec, ref):
    mse = ((rec.clamp(0, 1) - ref) ** 2).mean(dim=(1, 2, 3))
    return -10 * torch.log10(mse.clamp_min(1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-images", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-viz", type=int, default=4)
    ap.add_argument("--val-npz", default="data_eval/imagenet-256-val.npz")
    ap.add_argument("--k23-cfg", default="configs/stage1/sampling/dinov3l-k23-imagenet.yaml")
    ap.add_argument("--k7-cfg", default="configs/stage1/sampling/dinov3l-k7-imagenet.yaml")
    ap.add_argument("--ours-cfgs",
                    default="configs/stage1/decoder/infer-k23-fed-k7.yaml,"
                            "configs/stage1/decoder/infer-k23plain-fed-k7.yaml",
                    help="comma-sep list of our inference configs (each adds a column)")
    ap.add_argument("--out-png", default="output_full/feed_k7_compare.png")
    args = ap.parse_args()
    device = torch.device("cuda")
    mean, std = MEAN.to(device), STD.to(device)

    # A) official k7 (native 7-layer decoder)
    rae_k7 = instantiate_from_config(OmegaConf.load(args.k7_cfg).stage_1).to(device).eval()
    # B) official k23, but fed only the 7 layers (override layer selection)
    rae_k23 = instantiate_from_config(OmegaConf.load(args.k23_cfg).stage_1).to(device).eval()
    rae_k23.encoder.layer_indices = list(K7_LAYERS)
    print(f"official k23 encoder now using layers={rae_k23.encoder.layer_indices}", flush=True)

    # C) our k23 variants, fed the 7 layers via combine idx subset. One shared
    # DINOv3 encoder (layers 1..23); each config supplies its own combine + decoder.
    our_enc = create_encoder("dinov3mls-vit-l16[layers=" + ".".join(str(i) for i in range(1, 24)) + "]",
                             device=device, resolution=256).eval()
    for p in our_enc.parameters():
        p.requires_grad_(False)

    def make_ours(cfg_path):
        infer = OmegaConf.load(cfg_path)
        ck = torch.load(infer.eval.ckpt, map_location="cpu", weights_only=False)
        layers = list(ck["layers"])
        idx = [layers.index(L) for L in K7_LAYERS]
        combine = get_obj_from_str(infer.combine.target)(
            **OmegaConf.to_container(infer.combine.params, resolve=True)).to(device).eval()
        combine.load_state_dict(ck["ema_combine"])
        dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                            num_patches=256, pretrained_path=None).to(device).eval()
        dec.load_state_dict(ck["ema_dec"])
        tag = infer.eval.get("tag", os.path.basename(cfg_path).replace(".yaml", ""))
        del ck
        print(f"{tag}: trained layers={layers}, fed idx={idx}", flush=True)

        def recon(imgs01):
            toks = list(our_enc.model.get_intermediate_layers(
                (imgs01 - mean) / std, n=layers, reshape=False,
                return_class_token=False, norm=True))
            z = combine(toks, idx=idx)
            out = dec(z, drop_cls_token=False).logits
            return (dec.unpatchify(out) * std + mean)
        return tag, recon

    ours = [make_ours(c.strip()) for c in args.ours_cfgs.split(",") if c.strip()]

    arr = np.load(args.val_npz, mmap_mode="r")
    arr = arr[arr.files[0]] if hasattr(arr, "files") else arr
    g = torch.Generator().manual_seed(args.seed)
    idxs = torch.randperm(len(arr), generator=g)[:args.num_images].tolist()

    names = ["official_k7", "official_k23_fed7"] + [t for t, _ in ours]
    ncol = 1 + len(names)                                 # original + each model
    psnrs = {k: [] for k in names}
    ssims = {k: 0.0 for k in names}
    viz_rows, n_done = [], 0
    with torch.no_grad():
        for i in range(0, len(idxs), args.batch):
            bi = idxs[i:i + args.batch]
            imgs = torch.stack([torch.from_numpy(arr[j].copy()) for j in bi])
            imgs = imgs.permute(0, 3, 1, 2).float().to(device) / 255       # [0,1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rec = {"official_k7": rae_k7(imgs), "official_k23_fed7": rae_k23(imgs)}
                for tag, fn in ours:
                    rec[tag] = fn(imgs)
            for k in names:
                r = rec[k].clamp(0, 1).float()
                psnrs[k].append(per_image_psnr(r, imgs))
                ssims[k] += ssim_fn(r, imgs, data_range=1.0).item() * r.shape[0]
            if len(viz_rows) < args.n_viz:
                for b in range(imgs.shape[0]):
                    if len(viz_rows) >= args.n_viz:
                        break
                    row = [imgs[b].cpu()] + [rec[k][b].clamp(0, 1).float().cpu() for k in names]
                    viz_rows.append(torch.stack(row))
            n_done += imgs.shape[0]
            if n_done % 320 == 0:
                print(f"  {n_done}/{len(idxs)}", flush=True)

    print(f"\n=== Feed k=7 layers {K7_LAYERS} -> 3 decoders   (N={n_done} val images) ===")
    results = {}
    for k in names:
        p = torch.cat(psnrs[k]); s = ssims[k] / n_done
        results[k] = {"psnr": p.mean().item(), "psnr_std": p.std().item(), "ssim": s}
        print(f"  {k:20s}  PSNR {p.mean():6.3f} +/- {p.std():4.2f} dB   SSIM {s:.4f}")

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    grid = torch.cat(viz_rows, dim=0)                  # [n_viz*ncol, 3, 256, 256]
    save_image(grid, args.out_png, nrow=ncol)
    print(f"\nviz cols: original | " + " | ".join(names) + f"\n  -> {args.out_png}")
    with open(args.out_png.replace(".png", ".json"), "w") as f:
        json.dump({"k7_layers": K7_LAYERS, "num_images": n_done, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
