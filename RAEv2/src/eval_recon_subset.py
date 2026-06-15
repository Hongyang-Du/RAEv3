#!/usr/bin/env python3
"""Reconstruction PSNR + SSIM for a train_decoder.py (MLSCombine) stage-1 ckpt,
with an optional LAYER SUBSET so one trained model can be evaluated on fewer
layers than it was trained on.

Driven by a single YAML (plain OmegaConf, not the training DecoderConfig schema):

  combine:                      # arch to rebuild MLSCombine + load the ckpt weights
    target: stage1.combine.MLSCombine
    params: {layers: [...], weighting: ..., projector: ..., ...}
  eval:                         # optional block; CLI flags override
    ckpt: <path to ckpt>
    subset_layers: [11,13,...]  # feed ONLY these layer numbers (must be a subset of
                                #   the ckpt's trained layers); omit -> use all
    num_images: 1000
    val_npz: data_eval/imagenet-256-val.npz
    seed: 0
    tag: <name>

Examples:
  # k=23 drop+SIGReg model fed the raev2-default 7 layers (the inference experiment)
  python src/eval_recon_subset.py --config configs/stage1/decoder/infer-k23-fed-k7.yaml
  # k=7 original baseline (training config, all 7 layers)
  python src/eval_recon_subset.py --config configs/stage1/decoder/raev2-k7.yaml \
      --ckpt output_full/decoder_raev2_k7/ckpt_latest.pt --tag k7_original
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

from encoders.vision_encoder import create_encoder
from stage1.rae import _load_decoder
from utils.model_utils import get_obj_from_str

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def per_image_psnr(rec, ref):
    mse = ((rec.clamp(0, 1) - ref) ** 2).mean(dim=(1, 2, 3))
    return -10 * torch.log10(mse.clamp_min(1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="inference/training YAML (with combine block)")
    ap.add_argument("--ckpt", default=None, help="override eval.ckpt")
    ap.add_argument("--idx", default=None, help="override: comma-sep 0-based positions into layers")
    ap.add_argument("--num-images", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device("cuda")

    cfg = OmegaConf.load(args.config)
    ev = cfg.get("eval", {}) or {}
    ckpt_path = args.ckpt or ev.get("ckpt")
    if not ckpt_path:
        ap.error("no ckpt: pass --ckpt or set eval.ckpt in the config")
    num_images = args.num_images or ev.get("num_images", 1000)
    val_npz = ev.get("val_npz", "data_eval/imagenet-256-val.npz")
    seed = ev.get("seed", 0)
    batch = ev.get("batch", 32)
    tag = args.tag or ev.get("tag", "")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    layers = list(ck["layers"])

    # resolve subset -> 0-based positions into `layers`
    if args.idx:
        idx = [int(x) for x in args.idx.split(",")]
    elif ev.get("subset_layers"):
        subset = list(ev["subset_layers"])
        missing = [L for L in subset if L not in layers]
        if missing:
            raise ValueError(f"subset_layers {missing} not in ckpt layers {layers}")
        idx = [layers.index(L) for L in subset]
    else:
        idx = None
    used = [layers[i] for i in idx] if idx is not None else layers
    print(f"[{tag}] ckpt={os.path.basename(ckpt_path)} ep={ck.get('epoch')}\n"
          f"  trained layers={layers}\n  eval layers (idx={idx}) -> {used}", flush=True)

    enc = create_encoder("dinov3mls-vit-l16[layers=" + ".".join(map(str, layers)) + "]",
                         device=device, resolution=256).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    mean, std = MEAN.to(device), STD.to(device)

    params = OmegaConf.to_container(cfg.combine.params, resolve=True)
    combine = get_obj_from_str(cfg.combine.target)(**params).to(device).eval()
    combine.load_state_dict(ck["ema_combine"])
    dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                        num_patches=256, pretrained_path=None).to(device).eval()
    dec.load_state_dict(ck["ema_dec"])
    del ck

    arr = np.load(val_npz, mmap_mode="r")
    arr = arr[arr.files[0]] if hasattr(arr, "files") else arr      # [N,256,256,3] uint8
    g = torch.Generator().manual_seed(seed)
    idxs = torch.randperm(len(arr), generator=g)[:num_images].tolist()

    psnrs, ssims, n_done = [], [], 0
    with torch.no_grad():
        for i in range(0, len(idxs), batch):
            bi = idxs[i:i + batch]
            imgs = torch.stack([torch.from_numpy(arr[j].copy()) for j in bi])
            imgs = imgs.permute(0, 3, 1, 2).float().to(device) / 255    # [B,3,256,256]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                toks = list(enc.model.get_intermediate_layers(
                    (imgs - mean) / std, n=layers, reshape=False,
                    return_class_token=False, norm=True))
                z = combine(toks, idx=idx)
                out = dec(z, drop_cls_token=False).logits
                rec = dec.unpatchify(out) * std + mean
            rec = rec.clamp(0, 1).float()                  # metrics in fp32 (bf16 SSIM is wrong)
            psnrs.append(per_image_psnr(rec, imgs))
            ssims.append(ssim_fn(rec, imgs, data_range=1.0).item() * rec.shape[0])
            n_done += imgs.shape[0]
            if n_done % 320 == 0:
                print(f"  {n_done}/{len(idxs)}", flush=True)

    psnr = torch.cat(psnrs)
    ssim = sum(ssims) / n_done
    print(f"\n[{tag}] N={n_done}  eval_layers={used}")
    print(f"  PSNR = {psnr.mean():.3f} +/- {psnr.std():.3f} dB  (median {psnr.median():.3f})")
    print(f"  SSIM = {ssim:.4f}")

    out = args.out or ckpt_path.replace(".pt", f"_recon{('_'+tag) if tag else ''}.json")
    with open(out, "w") as f:
        json.dump({"ckpt": ckpt_path, "tag": tag, "eval_layers": used, "num_images": n_done,
                   "psnr_mean": psnr.mean().item(), "psnr_std": psnr.std().item(),
                   "ssim": ssim}, f, indent=2)
    print(f"json -> {out}")


if __name__ == "__main__":
    main()
