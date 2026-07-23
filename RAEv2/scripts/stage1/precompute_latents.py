#!/usr/bin/env python3
"""Precompute stage-1 latents for the FULL dataset (any stage_1 wrapper), so stage-2
DiT training reads cached tensors instead of re-running encoder+combine every step.

Only valid when the stage_1 wrapper's encode() is DETERMINISTIC for a given image
(e.g. RAECombine with drop: false -> combine always takes the full-mask branch).
If drop: true, encode() applies per-sample random layer-dropout and caching a single
pass per image would freeze that augmentation -- do not use this script for such a
config.

Output layout (mirrors nothing existing -- this is a new cache format, one manifest.json
+ N shard-*.pt files per split):
    <out-dir>/<split>/shard-r{rank:02d}-{idx:05d}.pt
        {"latents": bf16 [n,C,H,W] (already stats-normalized, matches rae.encode() output),
         "labels":  int64 [n],
         "indices": int64 [n]  (original dataset index, for provenance/debugging)}
    <out-dir>/<split>/manifest.json
        {"num_samples", "latent_shape", "dtype", "shard_size", "config", "stage1_ckpt_path",
         "normalization_stat_path", "shards": [{"file","rank","n"}...]}

Usage:
    torchrun --nproc_per_node=8 scripts/stage1/precompute_latents.py \
        --config configs/stage2/training/imagenet-dinov3l-depthattn-nano-p03-cls-k23.yaml \
        --data-dir /mnt/localssd/imagenet-256 \
        --split train \
        --out-dir /mnt/localssd/latents-depthattn-k23-nano-p03 \
        --shard-size 5000
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from compute_encoder_stats import setup_distributed, cleanup_distributed
from data import ImageNetHFDataset
from utils.model_utils import instantiate_from_config
from configs.stage2 import Stage2Config


class ShardWriter:
    """Buffers (latent, label, index) triples and flushes fixed-size .pt shards."""

    def __init__(self, out_dir: Path, rank: int, shard_size: int):
        self.out_dir = out_dir
        self.rank = rank
        self.shard_size = shard_size
        self.shard_idx = 0
        self.buf_z, self.buf_y, self.buf_idx = [], [], []
        self.manifest_entries = []

    def add(self, z: torch.Tensor, y: torch.Tensor, idx: torch.Tensor):
        self.buf_z.append(z)
        self.buf_y.append(y)
        self.buf_idx.append(idx)
        n = sum(t.shape[0] for t in self.buf_z)
        while n >= self.shard_size:
            self._flush(self.shard_size)
            n = sum(t.shape[0] for t in self.buf_z)

    def _flush(self, take: int):
        z = torch.cat(self.buf_z, dim=0)
        y = torch.cat(self.buf_y, dim=0)
        idx = torch.cat(self.buf_idx, dim=0)
        z_take, z_rest = z[:take], z[take:]
        y_take, y_rest = y[:take], y[take:]
        idx_take, idx_rest = idx[:take], idx[take:]
        fname = f"shard-r{self.rank:02d}-{self.shard_idx:05d}.pt"
        torch.save({"latents": z_take.contiguous(), "labels": y_take.contiguous(),
                    "indices": idx_take.contiguous()}, self.out_dir / fname)
        self.manifest_entries.append({"file": fname, "rank": self.rank, "n": int(z_take.shape[0])})
        self.shard_idx += 1
        self.buf_z = [z_rest] if z_rest.shape[0] else []
        self.buf_y = [y_rest] if y_rest.shape[0] else []
        self.buf_idx = [idx_rest] if idx_rest.shape[0] else []

    def close(self):
        if sum(t.shape[0] for t in self.buf_z) > 0:
            self._flush(sum(t.shape[0] for t in self.buf_z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     required=True, help="Stage-2 training YAML (stage_1 section is used)")
    ap.add_argument("--data-dir",   default="/mnt/localssd/imagenet-256")
    ap.add_argument("--split",      default="train")
    ap.add_argument("--out-dir",    required=True)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-samples", type=int, default=None, help="Subset size (default: full dataset); for smoke tests")
    args = ap.parse_args()

    rank, world_size, device, is_distributed = setup_distributed()

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config)))
    stat_path = config.stage_1.params.get("normalization_stat_path")
    if not stat_path or not Path(stat_path).exists():
        raise FileNotFoundError(
            f"normalization_stat_path missing/not found ({stat_path}) -- compute it first "
            f"(scripts/stage1/compute_latent_stats.py) so cached latents match what stage-2 training expects."
        )
    rae = instantiate_from_config(config.stage_1).to(device)
    rae.eval()
    if getattr(rae, "drop", False):
        raise ValueError("stage_1.params.drop is True -> encode() is stochastic per-sample; "
                          "precompute_latents.py only supports deterministic (drop: false) wrappers.")

    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])
    dataset = ImageNetHFDataset(data_dir=args.data_dir, split=args.split, transform=transform)
    total = len(dataset)
    if args.num_samples is not None and args.num_samples < total:
        total = args.num_samples
    my_indices = list(range(rank, total, world_size))
    subset = Subset(dataset, my_indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    if rank == 0:
        print(f"stage_1: {config.stage_1.target}  dataset: {total} samples ({args.split})")
        print(f"world_size={world_size}  shard_size={args.shard_size}  out_dir={out_dir}")

    writer = ShardWriter(out_dir, rank, args.shard_size)
    pos = 0
    for images, labels in tqdm(loader, disable=rank != 0, desc=f"encoding[{args.split}]"):
        with torch.no_grad():
            z = rae.encode(images.to(device)).to(torch.bfloat16).cpu()
        n = images.shape[0]
        idx = torch.tensor(my_indices[pos:pos + n], dtype=torch.long)
        pos += n
        labels = labels if torch.is_tensor(labels) else torch.tensor(labels)
        writer.add(z, labels.to(torch.long), idx)
    writer.close()

    if is_distributed:
        dist.barrier()
        gathered = [None] * world_size
        dist.all_gather_object(gathered, writer.manifest_entries)
    else:
        gathered = [writer.manifest_entries]

    if rank == 0:
        shards = [e for rank_entries in gathered for e in rank_entries]
        manifest = {
            "num_samples": total,
            "latent_shape": list(rae.encode(torch.zeros(1, 3, args.image_size, args.image_size, device=device)).shape[1:]),
            "dtype": "bfloat16",
            "shard_size": args.shard_size,
            "config": args.config,
            "stage1_ckpt_path": config.stage_1.params.get("stage1_ckpt_path"),
            "normalization_stat_path": stat_path,
            "shards": shards,
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        n_written = sum(e["n"] for e in shards)
        print(f"done: {n_written}/{total} samples -> {len(shards)} shards in {out_dir}")
        if n_written != total:
            print(f"WARNING: n_written ({n_written}) != dataset total ({total})")

    cleanup_distributed()


if __name__ == "__main__":
    main()
