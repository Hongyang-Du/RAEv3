#!/usr/bin/env python3
"""Semantic-degeneration linear probe across encoder->decoder depth.

For each depth we take the CLS token and train a linear ImageNet classifier on it
(frozen backbone). Curves on one shared depth axis:

  encoder      : DINOv3-L k7 layers 11,13,15,17,19 (drop last layer 21/23); cls token
                 per layer -> semantics RISE with depth.
  dec_ours     : our 16-epoch k23 decoder
  dec_k7       : official RAEv2 dinov3l-k7 decoder
  dec_k23      : official RAEv2 dinov3l-k23 decoder
                 (decoder cls token sampled every 2 transformer blocks -> semantics FALL)

Each decoder is fed the latent IT was trained on:
  - k7 decoder   <- mean over layers 11,13,15,17,19,21,23 + L23 cls surrogate
  - k23 / ours   <- mean over layers 1..23            + L23 cls surrogate

We read the decoder's own CLS token (position 0) after blocks 0,2,4,...; no image
reconstruction — we watch classification accuracy decay as the decoder trades
semantics for pixels. All heads train jointly from one frozen forward per batch.
Held-out top-1 on a fixed 25k train subset (no val split on this box).

Usage:
  CUDA_VISIBLE_DEVICES=0 python src/probe_semantic_degeneration.py \
      --data /mnt/localssd/imagenet-256-full --epochs 3 --batch-size 96 \
      --ours-ckpt output_full/decoder_xcong_plain_k23_continue16ep/ckpt_ep016.pt \
      --k7-decoder  <official dinov3l-k7/decoder.pt> \
      --k23-decoder <official dinov3l-k23/decoder.pt>
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from encoders.vision_encoder import create_encoder
from stage1.rae import _load_decoder
from stage1.combine import MLSCombine

K23_LAYERS = list(range(1, 24))                       # 1..23
K7_LAYERS = [11, 13, 15, 17, 19, 21, 23]              # official k7 feed
ENC_PROBE_LAYERS = [11, 13, 15, 17, 19, 21, 23]       # full k7 layers (incl. last) for the rising curve
DEC_BLOCK_STRIDE = 2                                  # probe decoder cls every 2 blocks


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
    ds_tr = PartialImageNetDataset(data_dir, split="train", transform=t_train)
    ds_va = PartialImageNetDataset(data_dir, split="train", transform=t_val)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds_tr), generator=g)
    va, tr = perm[:val_size].tolist(), perm[val_size:].tolist()
    return torch.utils.data.Subset(ds_tr, tr), torch.utils.data.Subset(ds_va, va)


def load_decoder_from_ckpt(path, device):
    dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                        num_patches=256, pretrained_path=None)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj["ema_dec"] if isinstance(obj, dict) and "ema_dec" in obj else obj
    missing, unexpected = dec.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [warn] {os.path.basename(path)}: missing={len(missing)} unexpected={len(unexpected)}")
    return dec.to(device).eval()


@torch.no_grad()
def decoder_cls_per_block(dec, z, block_ids):
    """z [B,N,1024] -> {block_id: cls token [B,1152]} for the requested blocks."""
    x = dec.decoder_embed(z)
    x = dec.interpolate_latent(x)
    cls = dec.trainable_cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = x + dec.decoder_pos_embed
    want = set(block_ids)
    out = {}
    for j, layer in enumerate(dec.decoder_layers):
        x = layer(x, head_mask=None)[0]
        if j in want:
            out[j] = x[:, 0, :]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/mnt/localssd/imagenet-256-full")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-size", type=int, default=25000)
    p.add_argument("--max-steps", type=int, default=0,
                   help="cap training steps per epoch (0 = full epoch); linear probes converge on a subset")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ours-ckpt", default="output_full/decoder_xcong_plain_k23_continue16ep/ckpt_ep016.pt")
    p.add_argument("--k7-decoder", required=True)
    p.add_argument("--k23-decoder", required=True)
    p.add_argument("--out-dir", default="output_full/semantic_probe")
    args = p.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # encoder loaded with all 23 layers (needed to build both k7 and k23 latents)
    encoder = create_encoder("dinov3mls-vit-l16[layers=" + ".".join(map(str, K23_LAYERS)) + "]",
                             device=device, resolution=args.image_size).eval()
    for q in encoder.parameters():
        q.requires_grad_(False)
    enc_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    enc_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # combines (eval mode -> mean over fed layers + L23 cls surrogate, no drop)
    combine_k23 = MLSCombine(layers=K23_LAYERS, weighting="random_drop", p_drop=0.3,
                             cls_surrogate=True, projector="none", dim=1024, out_dim=1024).to(device).eval()
    combine_k7 = MLSCombine(layers=K7_LAYERS, weighting="random_drop", p_drop=0.3,
                            cls_surrogate=True, projector="none", dim=1024, out_dim=1024).to(device).eval()

    dec_ours = load_decoder_from_ckpt(args.ours_ckpt, device)
    dec_k7 = load_decoder_from_ckpt(args.k7_decoder, device)
    dec_k23 = load_decoder_from_ckpt(args.k23_decoder, device)
    n_blocks = len(dec_ours.decoder_layers)
    block_ids = list(range(0, n_blocks, DEC_BLOCK_STRIDE))            # 0,2,4,...
    print(f"enc probe layers={ENC_PROBE_LAYERS}; decoder blocks probed={block_ids} (of {n_blocks})", flush=True)

    enc_keys = [f"enc_L{l}" for l in ENC_PROBE_LAYERS]
    ours_keys = [f"ours_b{j}" for j in block_ids]
    k7_keys = [f"k7_b{j}" for j in block_ids]
    k23_keys = [f"k23_b{j}" for j in block_ids]
    ALL = enc_keys + ours_keys + k7_keys + k23_keys
    ENC_DIM, DEC_DIM = 1024, 1152

    # index helpers for encoder cls extraction
    k23_idx = {l: i for i, l in enumerate(K23_LAYERS)}

    @torch.no_grad()
    def batch_features(imgs01):
        x = (imgs01 - enc_mean) / enc_std
        out = encoder.model.get_intermediate_layers(
            x, n=K23_LAYERS, reshape=False, return_class_token=True, norm=True)
        patch_all = [o[0] for o in out]                              # 23 x [B,N,1024]
        cls_all = [o[1] for o in out]                                # 23 x [B,1024]
        feats = {f"enc_L{l}": cls_all[k23_idx[l]].float() for l in ENC_PROBE_LAYERS}
        # latents: k7 decoder gets the 7-layer combine, ours/k23 get the 23-layer combine
        patch_k7 = [patch_all[k23_idx[l]] for l in K7_LAYERS]
        z_k7 = combine_k7(patch_k7, idx=None)
        z_k23 = combine_k23(patch_all, idx=None)
        for j, c in decoder_cls_per_block(dec_ours, z_k23, block_ids).items():
            feats[f"ours_b{j}"] = c.float()
        for j, c in decoder_cls_per_block(dec_k7, z_k7, block_ids).items():
            feats[f"k7_b{j}"] = c.float()
        for j, c in decoder_cls_per_block(dec_k23, z_k23, block_ids).items():
            feats[f"k23_b{j}"] = c.float()
        return feats

    class Heads(nn.Module):
        def __init__(self):
            super().__init__()
            self.h = nn.ModuleDict({
                k: nn.Sequential(nn.BatchNorm1d(ENC_DIM if k.startswith("enc") else DEC_DIM, affine=False),
                                 nn.Linear(ENC_DIM if k.startswith("enc") else DEC_DIM, 1000))
                for k in ALL})

        def forward(self, feats):
            return {k: self.h[k](feats[k]) for k in ALL}

    heads = Heads().to(device)
    opt = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=0.0)

    ds_tr, ds_va = make_datasets(args.data, args.image_size, args.val_size, args.seed)
    print(f"probe-train {len(ds_tr)}  probe-val {len(ds_va)}", flush=True)
    loader = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
    steps_per_epoch = min(len(loader), args.max_steps) if args.max_steps else len(loader)
    total_steps = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    history = []
    step = 0
    for epoch in range(args.epochs):
        heads.train()
        t0 = time.time()
        for bi, (imgs, labels) in enumerate(loader):
            if args.max_steps and bi >= args.max_steps:
                break
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast:
                feats = batch_features(imgs)
            logits = heads(feats)
            loss = sum(F.cross_entropy(v, labels) for v in logits.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"ep{epoch+1} s{step}/{total_steps} loss={loss.item():.1f} "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)

        heads.eval()
        correct = {k: 0 for k in ALL}
        seen = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast:
                    feats = batch_features(imgs)
                logits = heads(feats)
                for k in ALL:
                    correct[k] += (logits[k].argmax(1) == labels).sum().item()
                seen += labels.numel()
        accs = {k: correct[k] / seen * 100 for k in ALL}
        b0, bL = block_ids[0], block_ids[-1]
        print(f"[ep{epoch+1}] enc_L11={accs['enc_L11']:.1f} enc_L19={accs['enc_L19']:.1f} | "
              f"ours b{b0}={accs[f'ours_b{b0}']:.1f}->b{bL}={accs[f'ours_b{bL}']:.1f} | "
              f"k7 b{b0}={accs[f'k7_b{b0}']:.1f}->b{bL}={accs[f'k7_b{bL}']:.1f} | "
              f"k23 b{b0}={accs[f'k23_b{b0}']:.1f}->b{bL}={accs[f'k23_b{bL}']:.1f} ({time.time()-t0:.0f}s)", flush=True)
        history.append({"epoch": epoch + 1, **accs})
        with open(os.path.join(args.out_dir, "results.json"), "w") as f:
            json.dump({"enc_probe_layers": ENC_PROBE_LAYERS, "block_ids": block_ids, "n_blocks": n_blocks,
                       "enc_keys": enc_keys, "ours_keys": ours_keys, "k7_keys": k7_keys, "k23_keys": k23_keys,
                       "val_size": args.val_size, "history": history}, f, indent=2)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
