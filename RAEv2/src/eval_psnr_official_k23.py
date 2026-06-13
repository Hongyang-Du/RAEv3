#!/usr/bin/env python3
"""Proper rPSNR estimate for the official dinov3l-k23 decoder: N random
ImageNet images (val-style transform), per-image PSNR averaged — the protocol
the paper number uses (theirs: 50k ImageNet val; we only have train arrows,
so numbers are slightly optimistic)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from omegaconf import OmegaConf
from torchvision import transforms

from data.partial_imagenet import PartialImageNetDataset
from utils.model_utils import instantiate_from_config

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
device = torch.device("cuda")

cfg = OmegaConf.load("configs/stage1/sampling/dinov3l-k23-imagenet.yaml")
rae = instantiate_from_config(cfg.stage_1).to(device).eval()

tf = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])
ds = PartialImageNetDataset("/datasets/imagenet-256", split="train", transform=tf)
g = torch.Generator().manual_seed(0)
idxs = torch.randperm(len(ds), generator=g)[:N].tolist()
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idxs),
                                     batch_size=32, num_workers=8)

psnrs = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for imgs, _ in loader:
        imgs = imgs.to(device)
        rec = rae(imgs).float().clamp(0, 1)
        mse = ((rec - imgs) ** 2).mean(dim=(1, 2, 3))
        psnrs.append(-10 * torch.log10(mse.clamp_min(1e-10)))
psnrs = torch.cat(psnrs)
print(f"official k23 rPSNR over {len(psnrs)} ImageNet-train images: "
      f"{psnrs.mean():.2f} ± {psnrs.std():.2f} dB  (median {psnrs.median():.2f})")
