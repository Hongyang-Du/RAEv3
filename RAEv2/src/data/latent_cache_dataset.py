"""Reads latents precomputed by scripts/stage1/precompute_latents.py, so stage-2
training can skip the frozen encoder+combine forward pass every step.

Shard files are shuffled and split across (DDP rank, DataLoader worker) the same
way GenericWebDataset splits .tar shards; within a shard, samples are shuffled
too. Only valid when the shards were produced by a DETERMINISTIC stage_1.encode()
(e.g. RAECombine with drop: false) -- see precompute_latents.py's docstring.
"""
import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info


class LatentCacheDataset(IterableDataset):
    def __init__(self, latents_dir: str, split: str = "train", seed: int = 42):
        self.split_dir = Path(latents_dir) / split
        manifest_path = self.split_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no manifest.json at {manifest_path} -- run scripts/stage1/precompute_latents.py first"
            )
        manifest = json.loads(manifest_path.read_text())
        self.shard_files = [e["file"] for e in manifest["shards"]]
        self.num_samples = manifest["num_samples"]
        self.latent_shape = manifest["latent_shape"]
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        world_size, rank = 1, 0
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        split_factor = world_size * num_workers
        split_idx = rank * num_workers + worker_id

        shards = list(self.shard_files)
        random.Random(self.seed + self.epoch).shuffle(shards)
        my_shards = shards[split_idx::split_factor]

        for fname in my_shards:
            data = torch.load(self.split_dir / fname, map_location="cpu", weights_only=True)
            latents, labels = data["latents"], data["labels"]
            n = labels.shape[0]
            gen = torch.Generator().manual_seed(self.seed + self.epoch + (hash(fname) % 100_000))
            for i in torch.randperm(n, generator=gen).tolist():
                yield latents[i].float(), labels[i]
