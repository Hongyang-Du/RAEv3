#!/usr/bin/env python3
"""Per-layer usage (LOO + solo) for three of OUR k=23 decoders on the same val
images (all share one encoder + ImageNet preprocess; only training differs):
  - ours_raev2     : MEAN of 23 layers, NO drop, no projector  (brittle baseline)
  - ours_drop+SIGReg : random-drop MEAN -> BN-MLP
  - ours_drop_plain  : random-drop MEAN, no projector

For each layer i:  LOO dPSNR = PSNR(all 23) - PSNR(all but i)  (reliance on layer i)
                   solo PSNR = PSNR(layer i alone)              (sufficiency of layer i)

All three use combine(toks, idx) for subsets, so probing is identical across models.
Writes output_full/layer_usage_compare.png (+ .json).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from utils.model_utils import get_obj_from_str
from encoders.vision_encoder import create_encoder
from stage1.rae import _load_decoder

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def per_image_psnr(rec, ref):
    mse = ((rec.clamp(0, 1) - ref) ** 2).mean(dim=(1, 2, 3))
    return -10 * torch.log10(mse.clamp_min(1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-images", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-npz", default="data_eval/imagenet-256-val.npz")
    ap.add_argument("--raev2-cfg", default="configs/stage1/decoder/infer-raev2k23.yaml")
    ap.add_argument("--sigreg-cfg", default="configs/stage1/decoder/infer-k23-fed-k7.yaml")
    ap.add_argument("--plain-cfg", default="configs/stage1/decoder/infer-k23plain-fed-k7.yaml")
    ap.add_argument("--out", default="output_full/layer_usage_compare.png")
    args = ap.parse_args()
    device = torch.device("cuda")
    mean, std = MEAN.to(device), STD.to(device)

    LAYERS = list(range(1, 24))                 # 1..23
    K = len(LAYERS)

    # shared encoder (all 23 layers)
    enc = create_encoder("dinov3mls-vit-l16[layers=" + ".".join(map(str, LAYERS)) + "]",
                         device=device, resolution=256).eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    def encode(imgs01):
        return list(enc.model.get_intermediate_layers(
            (imgs01 - mean) / std, n=LAYERS, reshape=False,
            return_class_token=False, norm=True))

    # ---- our models: MEAN of subset (+ projector) -> our decoder ----
    def load_ours(cfg_path):
        infer = OmegaConf.load(cfg_path)
        ck = torch.load(infer.eval.ckpt, map_location="cpu", weights_only=False)
        combine = get_obj_from_str(infer.combine.target)(
            **OmegaConf.to_container(infer.combine.params, resolve=True)).to(device).eval()
        combine.load_state_dict(ck["ema_combine"])
        dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                            num_patches=256, pretrained_path=None).to(device).eval()
        dec.load_state_dict(ck["ema_dec"])
        del ck

        def fn(toks, idx):
            z = combine(toks, idx=list(idx))
            out = dec(z, drop_cls_token=False).logits
            return dec.unpatchify(out) * std + mean
        return fn

    models = {
        "ours_raev2": load_ours(args.raev2_cfg),
        "ours_drop+SIGReg": load_ours(args.sigreg_cfg),
        "ours_drop_plain": load_ours(args.plain_cfg),
    }

    # subsets: full, LOO_i, solo_i
    subsets = [list(range(K))] + [[j for j in range(K) if j != i] for i in range(K)] + \
              [[i] for i in range(K)]

    arr = np.load(args.val_npz, mmap_mode="r")
    arr = arr[arr.files[0]] if hasattr(arr, "files") else arr
    g = torch.Generator().manual_seed(args.seed)
    idxs = torch.randperm(len(arr), generator=g)[:args.num_images].tolist()

    sums = {k: torch.zeros(len(subsets), device=device) for k in models}
    n_done = 0
    with torch.no_grad():
        for i0 in range(0, len(idxs), args.batch):
            bi = idxs[i0:i0 + args.batch]
            imgs = torch.stack([torch.from_numpy(arr[j].copy()) for j in bi])
            imgs = imgs.permute(0, 3, 1, 2).float().to(device) / 255
            with torch.autocast("cuda", dtype=torch.bfloat16):
                toks = encode(imgs)
                for mk, fn in models.items():
                    for si, idx in enumerate(subsets):
                        sums[mk][si] += per_image_psnr(fn(toks, idx).float(), imgs).sum()
            n_done += imgs.shape[0]
            print(f"  {n_done}/{len(idxs)}", flush=True)

    res = {}
    for mk in models:
        m = (sums[mk] / n_done).tolist()
        full, loo, solo = m[0], m[1:1 + K], m[1 + K:]
        res[mk] = {"full": full, "loo_dpsnr": [full - v for v in loo], "solo": solo}
        print(f"[{mk}] full={full:.2f} dB")

    # ---- plot: LOO dPSNR (reliance) + solo PSNR (sufficiency) ----
    colors = {"ours_raev2": "tab:gray", "ours_drop+SIGReg": "tab:red", "ours_drop_plain": "tab:blue"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2))
    for mk in models:
        a1.plot(LAYERS, res[mk]["loo_dpsnr"], "o-", ms=4, color=colors[mk],
                label=f"{mk} (full {res[mk]['full']:.1f} dB)")
        a2.plot(LAYERS, res[mk]["solo"], "o-", ms=4, color=colors[mk], label=mk)
    a1.set_title("Leave-one-out  dPSNR  (reliance: higher = more dependent on that layer)")
    a1.set_xlabel("DINOv3 layer"); a1.set_ylabel("dPSNR vs full [dB]"); a1.grid(alpha=0.3); a1.legend(fontsize=8)
    a2.set_title("Solo  PSNR  (sufficiency: reconstruct from one layer alone)")
    a2.set_xlabel("DINOv3 layer"); a2.set_ylabel("PSNR [dB]"); a2.grid(alpha=0.3); a2.legend(fontsize=8)
    fig.suptitle(f"Per-layer usage of the 23 DINOv3 layers  (N={n_done} val images)", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"saved -> {args.out}")
    with open(args.out.replace(".png", ".json"), "w") as f:
        json.dump({"layers": LAYERS, "num_images": n_done, "results": res}, f, indent=2)


if __name__ == "__main__":
    main()
