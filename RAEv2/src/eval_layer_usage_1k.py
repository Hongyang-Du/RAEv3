#!/usr/bin/env python3
"""Unified layer-usage evaluation on 1000 random ImageNet images (paper-style
per-image PSNR, fixed seed 0 -> identical image set for every variant).

Computes full / LOO / solo PSNR for any of:
  official      official raev2 dinov3l-k23 (fixed mean + L23 surrogate, stats norm)
  dropmean_bn   ours: dropmean + BN projector ckpt   (eval = full mean)
  dropmean_ln   ours: dropmean + LN projector ckpt   (eval = full mean)
  softgate      ours: plain learnable softmax gate   (subset = renormalized gate)

Encoder features are extracted once per batch; each subset reuses them, so the
cost is ~(2K+1) decoder forwards per batch.

Usage (1 GPU, ~10 min for K=24):
    python src/eval_layer_usage_1k.py --variant official
    python src/eval_layer_usage_1k.py --variant dropmean_bn \
        --ckpt output_full/train_decoder_mls_dropmean_bn_all24/ckpt_latest.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torchvision import transforms

from data.partial_imagenet import PartialImageNetDataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def per_image_psnr(rec, ref):
    mse = ((rec.clamp(0, 1) - ref) ** 2).mean(dim=(1, 2, 3))
    return -10 * torch.log10(mse.clamp_min(1e-10))


def build_variant(args, device):
    """Returns (layers, encode_tokens(imgs01)->K x [B,N,C], decode_subset(toks, idx)->imgs01)."""
    v = args.variant
    if v == "official":
        from omegaconf import OmegaConf
        from utils.model_utils import instantiate_from_config
        cfg = OmegaConf.load("configs/stage1/sampling/dinov3l-k23-imagenet.yaml")
        rae = instantiate_from_config(cfg.stage_1).to(device).eval()
        layers = rae.encoder.layer_indices
        mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

        def encode_tokens(imgs01):
            return list(rae.encoder.model.get_intermediate_layers(
                (imgs01 - mean) / std, n=layers, reshape=False,
                return_class_token=False, norm=True))

        def decode_subset(toks, idx):
            z = torch.stack([toks[i] for i in idx]).mean(0) \
                + toks[-1].mean(dim=1, keepdim=True)        # FIXED L23 surrogate
            b, n, c = z.shape
            z = z.transpose(1, 2).view(b, c, 16, 16)
            if rae.do_normalization:
                z = (z - rae.latent_mean.to(device)) / torch.sqrt(rae.latent_var.to(device) + rae.eps)
            return rae.decode(z)

        return layers, encode_tokens, decode_subset

    # ---- our variants: frozen DINOv3 + ckpt(ema_proj?, ema_dec) ----------------
    assert args.ckpt, f"--ckpt required for variant {v}"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    layers = ck["layers"]
    print(f"{v}: ckpt epoch {ck.get('epoch')}, layers={layers}")

    from encoders.vision_encoder import create_encoder
    enc = create_encoder("dinov3mls-vit-l16[layers=" + ".".join(map(str, layers)) + "]",
                         device=device, resolution=256)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    from stage1.rae import _load_decoder
    dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                        num_patches=256, pretrained_path=None).to(device).eval()
    dec.load_state_dict(ck["ema_dec"])

    def encode_tokens(imgs01):
        return list(enc.model.get_intermediate_layers(
            (imgs01 - mean) / std, n=layers, reshape=False,
            return_class_token=False, norm=True))

    if v == "raev2_ours":
        # our retrained raev2 baseline (train_decoder_mls.py): NO projector;
        # combine = subset mean + FIXED last-layer token-mean surrogate
        def decode_subset(toks, idx):
            z = torch.stack([toks[i] for i in idx]).mean(0) \
                + toks[-1].mean(dim=1, keepdim=True)
            out = dec(z, drop_cls_token=False).logits
            return dec.unpatchify(out) * std + mean
        return layers, encode_tokens, decode_subset

    if v == "dropmean_plain":
        # our random-drop, projector=none, NO cls surrogate (decoder_random_drop_layer
        # _mls_plain_k23): combine = equal-weight renormalized subset mean -> decoder.
        def decode_subset(toks, idx):
            z = torch.stack([toks[i] for i in idx]).mean(0)
            out = dec(z, drop_cls_token=False).logits
            return dec.unpatchify(out) * std + mean
        return layers, encode_tokens, decode_subset

    if v == "dropmean_bn":
        from train_decoder_mls_dropmean_bn_sigreg import MLSProjector
        proj = MLSProjector(dim=enc.hidden_size, out_dim=1024)
    elif v == "dropmean_ln":
        from train_decoder_mls_dropmean_sigreg import MLSProjector
        proj = MLSProjector(dim=enc.hidden_size, out_dim=1024)
    elif v == "nogate":
        from train_decoder_mls_nogate_sigreg import MLSProjector
        proj = MLSProjector(dim=enc.hidden_size, out_dim=1024)
    elif v == "softgate":
        from train_decoder_mls_softgate_plain import MLSProjector
        proj = MLSProjector(n_layers=len(layers), dim=enc.hidden_size, out_dim=1024)
    else:
        raise ValueError(v)
    proj.load_state_dict(ck["ema_proj"])
    proj.to(device).eval()
    for p in proj.parameters():
        p.requires_grad_(False)
    gate = (v == "softgate")

    def decode_subset(toks, idx):
        sub = [toks[i] for i in idx]
        z = proj(sub, idx=idx) if gate else proj(sub)
        out = dec(z, drop_cls_token=False).logits
        return dec.unpatchify(out) * std + mean              # our decoders: de-normalize

    return layers, encode_tokens, decode_subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["official", "raev2_ours", "nogate",
                             "dropmean_plain", "dropmean_bn", "dropmean_ln", "softgate"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--num-images", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--data", default="/datasets/imagenet-256")
    ap.add_argument("--val-npz", default="data_eval/imagenet-256-val.npz",
                    help="official 50k val npz; used when it exists (paper protocol), "
                         "else falls back to random train images")
    ap.add_argument("--seed", type=int, default=0, help="same seed -> same images for all variants")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device("cuda")

    layers, encode_tokens, decode_subset = build_variant(args, device)
    K = len(layers)

    g = torch.Generator().manual_seed(args.seed)
    if args.val_npz and os.path.exists(args.val_npz):
        import numpy as np

        class NpzVal(torch.utils.data.Dataset):
            def __init__(self, path):
                z = np.load(path, mmap_mode="r")
                self.arr = z[z.files[0]]                     # [N,256,256,3] uint8

            def __len__(self):
                return len(self.arr)

            def __getitem__(self, i):
                x = torch.from_numpy(self.arr[i].copy()).permute(2, 0, 1).float() / 255
                return x, 0

        ds = NpzVal(args.val_npz)
        split_used = f"val-npz ({len(ds)} images)"
    else:
        tf = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
        ds = PartialImageNetDataset(args.data, split="train", transform=tf)
        split_used = "train-arrows (val npz not found)"
    print(f"eval split: {split_used}")
    idxs = torch.randperm(len(ds), generator=g)[:args.num_images].tolist()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idxs),
                                         batch_size=args.batch, num_workers=8)

    subsets = [list(range(K))] + \
              [[j for j in range(K) if j != i] for i in range(K)] + \
              [[i] for i in range(K)]
    sums = torch.zeros(len(subsets), device=device)
    n_done = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for imgs, _ in loader:
            imgs = imgs.to(device)
            toks = encode_tokens(imgs)
            for si, idx in enumerate(subsets):
                sums[si] += per_image_psnr(decode_subset(toks, idx).float(), imgs).sum()
            n_done += imgs.shape[0]
            if n_done % 160 == 0:
                print(f"  {n_done}/{args.num_images}", flush=True)

    means = (sums / n_done).tolist()
    full, loo, solo = means[0], means[1:1 + K], means[1 + K:]
    dloo = [full - v for v in loo]
    print(f"[{args.variant}] full PSNR = {full:.2f} dB  ({n_done} images, per-image mean)")
    print("LOO dPSNR = [" + " ".join(f"{v:+.2f}" for v in dloo) + f"]  layers={layers}")
    print("solo PSNR = [" + " ".join(f"{v:.2f}" for v in solo) + "]")

    out = args.out or f"output_full/layer_usage_1k_{args.variant}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"variant": args.variant, "ckpt": args.ckpt, "layers": layers,
                   "num_images": n_done, "seed": args.seed, "split": split_used,
                   "full": full, "loo_dpsnr": dloo, "solo": solo}, f, indent=2)
    print(f"json -> {out}")


if __name__ == "__main__":
    main()
