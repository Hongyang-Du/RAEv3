#!/usr/bin/env python3
"""Per-layer usage probe for the OFFICIAL raev2 dinov3l-k23 stage-1 decoder
(nyu-visionx/RAEv2-models, layers 1..23, mean + L23 token-mean CLS surrogate).

Computes on the 5 fixed val images (assets/samples, Resize288+CenterCrop256 —
same protocol as all our stage-1 probes):
  full PSNR             : official combine over all 23 layers
  LOO dPSNR (23 values) : full − (subset mean without layer i, surrogate kept)
  solo PSNR (23 values) : layer i alone + surrogate

Convention matches our earlier raev2-K7 probe: the CLS surrogate stays the
FIXED L23 token-mean for every subset. Latent stats (stats.pt) are applied on
encode and inverted in decode, exactly as the stock RAE does.

Usage (any GPU with ~8 GB free):
    CUDA_VISIBLE_DEVICES=4 python src/probe_official_k23.py
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from overfit_sigreg import psnr
from utils.model_utils import instantiate_from_config

CONFIG = "configs/stage1/sampling/dinov3l-k23-imagenet.yaml"
VAL_DIR = "assets/samples"
OUT_JSON = "output_full/layer_usage_official_k23.json"
OUT_PNG = "output_full/layer_usage_official_k23.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = OmegaConf.load(CONFIG)
rae = instantiate_from_config(cfg.stage_1).to(device).eval()
layers = rae.encoder.layer_indices
print(f"Official k23 decoder loaded; layers={layers}")

# -- fixed val images ----------------------------------------------------------
t = transforms.Compose([
    transforms.Resize(256 + 32),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])
paths = sorted(p for p in (glob.glob(os.path.join(VAL_DIR, "*.png"))
                           + glob.glob(os.path.join(VAL_DIR, "*.jpg")))
               if "concat" not in os.path.basename(p).lower())
imgs = torch.stack([t(Image.open(p).convert("RGB")) for p in paths]).to(device)
print(f"{len(paths)} val images from {VAL_DIR}")

enc_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
enc_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

with torch.no_grad():
    toks = list(rae.encoder.model.get_intermediate_layers(
        (imgs - enc_mean) / enc_std, n=layers, reshape=False,
        return_class_token=False, norm=True))                  # 23 x [B,256,1024]
    surrogate = toks[-1].mean(dim=1, keepdim=True)             # FIXED L23 token-mean

    def decode_subset(idx):
        z = torch.stack([toks[i] for i in idx]).mean(0) + surrogate   # [B,256,1024]
        b, n, c = z.shape
        z = z.transpose(1, 2).view(b, c, 16, 16)
        if rae.do_normalization:
            z = (z - rae.latent_mean.to(device)) / torch.sqrt(rae.latent_var.to(device) + rae.eps)
        return psnr(rae.decode(z).clamp(0, 1), imgs)

    K = len(layers)
    full = decode_subset(list(range(K)))
    # sanity: the stock pipeline should give the same number
    full_stock = psnr(rae(imgs).clamp(0, 1), imgs)
    loo = [decode_subset([j for j in range(K) if j != i]) for i in range(K)]
    solo = [decode_subset([i]) for i in range(K)]

dloo = [full - v for v in loo]
print(f"Full PSNR = {full:.2f} dB  (stock pipeline sanity: {full_stock:.2f})")
print("LOO dPSNR = [" + " ".join(f"{v:+.2f}" for v in dloo) + f"]  layers={layers}")
print("solo PSNR = [" + " ".join(f"{v:.2f}" for v in solo) + "]")

os.makedirs("output_full", exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump({"layers": layers, "full": full, "full_stock": full_stock,
               "loo_dpsnr": dloo, "solo": solo}, f, indent=2)
print(f"json -> {OUT_JSON}")

# -- figure ---------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(K)
cmap = plt.get_cmap("viridis")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.bar(x, dloo, color=[cmap(i / (K - 1)) for i in range(K)])
ax1.axhline(0, color="k", lw=0.8)
ax1.set_xticks(x[::2]); ax1.set_xticklabels([f"L{l}" for l in layers[::2]])
ax1.set_ylabel("LOO ΔPSNR [dB]  (full − without layer)")
ax1.set_title("Reliance: official k23 decoder")
ax1.grid(alpha=0.3, axis="y")
ax2.plot(x, solo, "o-", color="tab:purple", lw=2, ms=5)
ax2.axhline(full, color="tab:red", ls="--", lw=1.5, label=f"full (23 layers): {full:.2f} dB")
ax2.set_xticks(x[::2]); ax2.set_xticklabels([f"L{l}" for l in layers[::2]])
ax2.set_ylabel("solo PSNR [dB]  (layer + L23 surrogate)")
ax2.set_title("Sufficiency: each layer alone")
ax2.legend(); ax2.grid(alpha=0.3)
fig.suptitle("OFFICIAL raev2 dinov3l-k23 decoder — per-layer usage "
             "(fixed mean combine + L23 CLS surrogate)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"figure -> {OUT_PNG}")
