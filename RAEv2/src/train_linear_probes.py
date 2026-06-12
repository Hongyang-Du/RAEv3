#!/usr/bin/env python3
"""Linear-probe ImageNet classification for ALL latent variants at once.

One frozen DINOv3-L forward per batch feeds 11 linear heads simultaneously:
  L11..L23      - each of the 7 DINOv3 layers alone (does the middle of the
                  trunk carry semantics?)
  mls_mean      - plain equal-weight mean over the 7 layers (projector INPUT)
  raev2         - raev2 MLS combine (mean + L23 token-mean CLS surrogate)
  nogate        - frozen EMA projector output of the nogate+SIGReg stage-1
  dropmean      - frozen EMA projector output of the dropmean+SIGReg stage-1

Feature = patch-token average pool [1024]; head = BatchNorm1d(affine=False)
-> Linear(1024, 1000) so different feature scales (SIGReg latents are ~N(0,1),
raw DINOv3 tokens are not) get a fair optimization.

No ImageNet val split on this box -> a fixed random 25k subset of train
(seed 42) is held out for top-1 eval; the rest is probe-train. Identical
data/transforms for every head, heads are independent (losses summed).

Usage:
    torchrun --nproc_per_node=8 src/train_linear_probes.py \
        --data /datasets/imagenet-256-full --epochs 5 --batch-size 128
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

from stage1.rae_variants import MLSProjector

LAYERS = [11, 13, 15, 17, 19, 21, 23]
PROBES = [f"L{l}" for l in LAYERS] + ["mls_mean", "raev2", "nogate", "dropmean"]


def make_datasets(data_dir, image_size, val_size, seed):
    t_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    t_val = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    from data.partial_imagenet import PartialImageNetDataset
    ds_train_full = PartialImageNetDataset(data_dir, split="train", transform=t_train)
    ds_val_full = PartialImageNetDataset(data_dir, split="train", transform=t_val)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds_train_full), generator=g)
    val_idx, train_idx = perm[:val_size].tolist(), perm[val_size:].tolist()
    return (torch.utils.data.Subset(ds_train_full, train_idx),
            torch.utils.data.Subset(ds_val_full, val_idx))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        required=True)
    p.add_argument("--image-size",  type=int, default=256)
    p.add_argument("--epochs",      type=int, default=5)
    p.add_argument("--batch-size",  type=int, default=128, help="per GPU")
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--val-size",    type=int, default=25000)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--nogate-ckpt",
                   default="output_full/train_decoder_mls_nogate_sigreg/ckpt_latest.pt")
    p.add_argument("--dropmean-ckpt",
                   default="output_full/train_decoder_mls_dropmean_sigreg/ckpt_latest.pt")
    p.add_argument("--out-dir",     default="output_full/linear_probes")
    args = p.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    is_main = rank == 0
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"Probes: {PROBES}\nWorld {world}, batch {args.batch_size}/GPU "
              f"(global {args.batch_size * world}), {args.epochs} epochs", flush=True)

    # -- frozen encoder + frozen EMA projectors --------------------------------
    from encoders.vision_encoder import create_encoder
    layers_str = ".".join(str(l) for l in LAYERS)
    encoder = create_encoder(f"dinov3mls-vit-l16[layers={layers_str}]",
                             device=device, resolution=args.image_size)
    encoder.eval()
    for q in encoder.parameters():
        q.requires_grad_(False)

    projs = {}
    for name, ckpt_path in (("nogate", args.nogate_ckpt), ("dropmean", args.dropmean_ckpt)):
        proj = MLSProjector(dim=encoder.hidden_size, out_dim=1024, mult=4)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        proj.load_state_dict(ck["ema_proj"])
        proj.to(device).eval()
        for q in proj.parameters():
            q.requires_grad_(False)
        projs[name] = proj
        if is_main:
            print(f"{name}: loaded ema_proj from {ckpt_path} (epoch {ck.get('epoch')})", flush=True)
        del ck

    enc_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    enc_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def batch_features(imgs01):
        """[B,3,H,W] in [0,1] -> dict probe_name -> [B, 1024] pooled feature."""
        x = (imgs01 - enc_mean) / enc_std
        layer_tokens = list(encoder.model.get_intermediate_layers(
            x, n=encoder.layer_indices, reshape=False,
            return_class_token=False, norm=True))                  # 7 x [B,N,1024]
        stk = torch.stack(layer_tokens)                            # [7,B,N,1024]
        feats = {f"L{l}": stk[i].mean(1) for i, l in enumerate(LAYERS)}
        z0 = stk.mean(0)                                           # [B,N,1024]
        feats["mls_mean"] = z0.mean(1)
        feats["raev2"] = (z0 + layer_tokens[-1].mean(1, keepdim=True)).mean(1)
        for name, proj in projs.items():
            feats[name] = proj(layer_tokens).mean(1)
        return {k: v.float() for k, v in feats.items()}

    # -- heads ------------------------------------------------------------------
    class ProbeHeads(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = nn.ModuleDict({
                name: nn.Sequential(nn.BatchNorm1d(1024, affine=False),
                                    nn.Linear(1024, 1000))
                for name in PROBES
            })

        def forward(self, feats):
            return {k: self.heads[k](feats[k]) for k in PROBES}

    heads = ProbeHeads().to(device)
    heads_ddp = DDP(heads, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=0.0)

    ds_train, ds_val = make_datasets(args.data, args.image_size, args.val_size, args.seed)
    if is_main:
        print(f"probe-train {len(ds_train)}  probe-val {len(ds_val)}", flush=True)
    sampler = torch.utils.data.distributed.DistributedSampler(
        ds_train, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    loader = torch.utils.data.DataLoader(
        ds_train, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        ds_val, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    total_steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    history = []
    step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        heads_ddp.train()
        t0 = time.time()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast:
                feats = batch_features(imgs)
            logits = heads_ddp(feats)
            losses = {k: F.cross_entropy(v, labels) for k, v in logits.items()}
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if is_main and step % 100 == 0:
                msg = "  ".join(f"{k}={losses[k].item():.2f}" for k in
                                ("L11", "L17", "L23", "mls_mean", "raev2", "nogate", "dropmean"))
                print(f"ep{epoch+1} s{step}/{total_steps}  {msg}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        # -- eval -----------------------------------------------------------
        heads_ddp.eval()
        correct = {k: torch.zeros((), device=device) for k in PROBES}
        seen = torch.zeros((), device=device)
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast:
                    feats = batch_features(imgs)
                logits = heads(feats)
                for k in PROBES:
                    correct[k] += (logits[k].argmax(1) == labels).sum()
                seen += labels.numel()
        for k in PROBES:
            dist.all_reduce(correct[k])
        dist.all_reduce(seen)
        accs = {k: (correct[k] / seen).item() * 100 for k in PROBES}
        if is_main:
            print(f"[Epoch {epoch+1}] val top-1 (%):  " +
                  "  ".join(f"{k}={accs[k]:.2f}" for k in PROBES) +
                  f"   ({time.time()-t0:.0f}s)", flush=True)
            history.append({"epoch": epoch + 1, **accs})
            with open(os.path.join(args.out_dir, "results.json"), "w") as f:
                json.dump({"probes": PROBES, "layers": LAYERS,
                           "val_size": args.val_size, "history": history}, f, indent=2)
            torch.save(heads.state_dict(), os.path.join(args.out_dir, "heads_latest.pt"))

    dist.destroy_process_group()
    if is_main:
        print("Done.", flush=True)


if __name__ == "__main__":
    main()
