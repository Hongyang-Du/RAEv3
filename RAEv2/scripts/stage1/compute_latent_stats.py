#!/usr/bin/env python3
"""Compute latent mean/var stats for ANY stage-1 wrapper via its own encode().

Unlike compute_encoder_stats.py (which re-implements the stock RAE encoding), this
instantiates the stage_1 section of a stage-2 training config — including our
RAEMLSBaseline / RAEProjected variants whose latent is a projector output — nulls its
normalization_stat_path, and aggregates Welford stats of rae.encode(images) over the
training set with the same Resize+ToTensor pipeline stage-2 training uses.

Usage (inside rae container):
    torchrun --nproc_per_node=8 scripts/stage1/compute_latent_stats.py \
        --config configs/stage2/training/imagenet-dinov3l-k7-nogate-sigreg.yaml \
        --data-dir /datasets/imagenet-256-full \
        --output-path output_full/train_decoder_mls_nogate_sigreg/latent_stats.pt \
        --num-samples 250000
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from compute_encoder_stats import WelfordAggregator, setup_distributed, cleanup_distributed
from data import ImageNetHFDataset
from utils.model_utils import instantiate_from_config
from configs.stage2 import Stage2Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      required=True, help="Stage-2 training YAML (stage_1 section is used)")
    ap.add_argument("--data-dir",    default="/datasets/imagenet-256-full")
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--image-size",  type=int, default=256)
    ap.add_argument("--batch-size",  type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-samples", type=int, default=None, help="Subset size (default: full dataset)")
    args = ap.parse_args()

    rank, world_size, device, is_distributed = setup_distributed()

    config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config)))
    config.stage_1.params["normalization_stat_path"] = None     # we are computing these
    rae = instantiate_from_config(config.stage_1).to(device)
    rae.eval()

    # same pipeline stage-2 training feeds rae.encode (hf dataset default transform)
    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    dataset = ImageNetHFDataset(data_dir=args.data_dir, split="train", transform=transform)
    if args.num_samples is not None and args.num_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(args.num_samples)))

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=False, drop_last=False) if is_distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    h = w = int((rae.base_patches) ** 0.5)
    aggregator = WelfordAggregator((rae.latent_dim, h, w), device=device)

    if rank == 0:
        print(f"stage_1: {config.stage_1.target}  latent: [{rae.latent_dim}, {h}, {w}]")
        print(f"samples: {len(dataset)}  batches/GPU: {len(loader)}")

    for images, _ in tqdm(loader, disable=rank != 0, desc="encoding"):
        with torch.no_grad():
            z = rae.encode(images.to(device))
        aggregator.update_batch(z)

    if is_distributed:
        aggregator.all_reduce()
    mean, var = aggregator.finalize()

    if rank == 0:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": mean.cpu(), "var": var.cpu()}, out)
        print(f"n={aggregator.n}")
        print(f"mean: range [{mean.min():.4f}, {mean.max():.4f}]  |mean|.mean={mean.abs().mean():.4f}")
        print(f"var:  range [{var.min():.4f}, {var.max():.4f}]  var.mean={var.mean():.4f}")
        print(f"(SIGReg sanity: |mean|~0, var~1 means the latent is already ~N(0,I))")
        print(f"saved -> {out}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
