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
from torch.utils.data import IterableDataset, get_worker_info


class LatentCacheDataset(IterableDataset):
    def __init__(self, latents_dir: str, split: str = "train", seed: int = 42,
                 rank: int = 0, world_size: int = 1):
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
        # Capture (rank, world_size) HERE in the main process, where the DDP process group
        # is initialized. The DataLoader spawns workers via forkserver, which do NOT inherit
        # the process group, so querying torch.distributed inside __iter__ (the worker) would
        # see world_size=1 -> every rank iterates the FULL shard set instead of its 1/world
        # slice: identical data on all ranks (effective batch collapses to one rank's) and a
        # ~world_size-times-too-long epoch that never ends (so epoch-boundary ckpts never fire).
        self.rank = rank
        self.world_size = world_size

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        # Use the (rank, world_size) captured at construction (see __init__): the process
        # group is unavailable in forkserver workers, so we must NOT read torch.distributed
        # here or sharding silently degrades to world_size=1.
        world_size, rank = self.world_size, self.rank
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        split_factor = world_size * num_workers
        split_idx = rank * num_workers + worker_id

        shards = list(self.shard_files)
        random.Random(self.seed + self.epoch).shuffle(shards)
        my_shards = shards[split_idx::split_factor]

        # Yield EXACTLY `budget` samples per (rank, worker) so every DDP rank runs an
        # identical number of steps -> no all-reduce desync at epoch end. Shards are uneven
        # in both count-per-rank (264 not divisible by world*workers) and size (bf16 shards
        # range from tens of MB to GBs), so iterating my_shards to exhaustion would give
        # ranks different lengths. Cap at the fair share and cycle this slot's shards (with a
        # fresh order/seed each pass) if they're short of it. Mirrors how the wds loader pins
        # a fixed epoch length via .with_epoch(steps).
        budget = self.num_samples // split_factor
        yielded = 0
        cycle = 0
        while yielded < budget and my_shards:
            order = list(my_shards)
            random.Random(self.seed + self.epoch + 1_000_003 * (cycle + 1)).shuffle(order)
            for fname in order:
                data = torch.load(self.split_dir / fname, map_location="cpu", weights_only=True)
                latents, labels = data["latents"], data["labels"]
                n = labels.shape[0]
                gen = torch.Generator().manual_seed(self.seed + self.epoch + cycle + (hash(fname) % 100_000))
                for i in torch.randperm(n, generator=gen).tolist():
                    if yielded >= budget:
                        return
                    yield latents[i].float(), labels[i]
                    yielded += 1
            cycle += 1
