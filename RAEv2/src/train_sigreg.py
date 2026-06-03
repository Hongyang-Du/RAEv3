#!/usr/bin/env python3
"""
Stage-1 training: learn SIGReg representation from DINOv3 register tokens.

Architecture:
  DINOv3 (frozen) → register tokens [B, 4, 1024]
       ↓  MLPProjector (learned)
  z_reg [B, 4, 1024]   ← SIGReg applied here (needs batch >> 1)
       ↓  TokenUpsampler  (4 → 256 via cross-attention)
  z_up  [B, 256, 1024]
       ↓  ViT Decoder (fine-tuned from pretrained)
  x_rec [B, 3, 256, 256]

Loss: L1 + LPIPS + λ_sig * SIGReg

Usage:
    cd RAEv2
    python src/train_sigreg.py \
        --data /home/colligo/data/imagenette2-320 \
        --epochs 20 --batch-size 64
"""

import sys, os, math, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
import wandb as wb

# reuse modules from overfit script
from overfit_sigreg import (MLPProjector, TokenUpsampler, SpatialAttnRes,
                            sigreg_loss, all_gather_with_grad, load_lpips)


# ─── Data ────────────────────────────────────────────────────────────────────

def make_loader(data_dir, split, image_size, batch_size, num_workers=4):
    t = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.RandomCrop(image_size) if split == "train" else transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip() if split == "train" else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
    ])
    root = os.path.join(data_dir, split)
    ds   = datasets.ImageFolder(root, transform=t)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split=="train"),
                      num_workers=num_workers, pin_memory=True, drop_last=True)


# ─── EMA ─────────────────────────────────────────────────────────────────────

def update_ema(ema_model, model, decay=0.9995):
    with torch.no_grad():
        for ep, p in zip(ema_model.parameters(), model.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1 - decay)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        required=True, help="ImageFolder root (train/val splits)")
    parser.add_argument("--image-size",  type=int, default=256)
    parser.add_argument("--epochs",      type=int, default=20)
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--sigreg-w",    type=float, default=0.1)
    parser.add_argument("--lpips-w",     type=float, default=1.0)
    parser.add_argument("--ema-decay",   type=float, default=0.9995)
    parser.add_argument("--log-every",   type=int, default=50)
    parser.add_argument("--val-every",   type=int, default=500)
    parser.add_argument("--out-dir",     default="output/train_sigreg")
    parser.add_argument("--ckpt-every",  type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--wandb",       action="store_true")
    parser.add_argument("--wandb-project", default="rae")
    parser.add_argument("--wandb-entity",  default="hongyangd")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  batch={args.batch_size}  |  epochs={args.epochs}")

    # ── wandb ─────────────────────────────────────────────────────────────────
    if args.wandb:
        wb.init(project=args.wandb_project, entity=args.wandb_entity,
                name="sigreg-stage1", config=vars(args),
                tags=["sigreg", "register-tokens", "stage1"])

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader = make_loader(args.data, "train", args.image_size,
                               args.batch_size, args.num_workers)
    val_loader   = make_loader(args.data, "val",   args.image_size,
                               min(args.batch_size, 32), args.num_workers)
    print(f"Train: {len(train_loader.dataset)} images  "
          f"Val: {len(val_loader.dataset)} images")

    # ── DINOv3 encoder (frozen) ───────────────────────────────────────────────
    from encoders.vision_encoder import create_encoder
    encoder = create_encoder("dinov3-vit-l16", device=device, resolution=args.image_size)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    enc_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    enc_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    # ── Decoder (fine-tune from pretrained) ───────────────────────────────────
    from omegaconf import OmegaConf
    from stage1.rae import _load_decoder
    cfg = OmegaConf.load("configs/stage2/training/imagenet-dinov3l-k7.yaml")
    s1  = cfg.stage_1.params
    decoder = _load_decoder(
        s1.decoder_config_path, hidden_size=1024, patch_size=16,
        num_patches=256, pretrained_path=s1.pretrained_decoder_path,
    ).to(device)

    # ── Learnable modules ──────────────────────────────────────────────────────
    projector  = MLPProjector(dim=1024).to(device)
    upsampler  = TokenUpsampler(n_out=256, dim=1024, n_heads=8, depth=3).to(device)

    # EMA copies
    from copy import deepcopy
    ema_proj = deepcopy(projector); ema_proj.requires_grad_(False)
    ema_ups  = deepcopy(upsampler);  ema_ups.requires_grad_(False)

    trainable = (list(projector.parameters()) +
                 list(upsampler.parameters()) +
                 list(decoder.parameters()))
    print(f"Trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    # ── LPIPS ─────────────────────────────────────────────────────────────────
    lpips_fn = load_lpips(device)

    # ── Training ──────────────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(args.epochs):
        projector.train(); upsampler.train(); decoder.train()
        epoch_losses = []
        t0 = time.time()

        for step, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)   # [B, 3, H, W] in [0,1]

            # Encode with DINOv3 (frozen)
            imgs_norm = (imgs - enc_mean) / enc_std
            with torch.no_grad():
                feats     = encoder.model.forward_features(imgs_norm)
                reg_tokens = feats["x_storage_tokens"]   # [B, 4, 1024]

            optimizer.zero_grad(set_to_none=True)

            # Forward
            z_reg = projector(reg_tokens)     # [B, 4, 1024]
            z_up  = upsampler(z_reg)          # [B, 256, 1024]

            dec_out = decoder(z_up, drop_cls_token=False).logits
            x_rec   = decoder.unpatchify(dec_out)
            x_rec   = (x_rec * enc_std + enc_mean).clamp(0, 1)

            # Losses
            loss_l1   = F.l1_loss(x_rec, imgs)
            loss_lpips = (lpips_fn(x_rec * 2 - 1, imgs * 2 - 1)
                          if lpips_fn else torch.zeros(1, device=device))
            # SIGReg: flatten [B, 4, 1024] → [B*4, 1024], batch > 1 needed
            loss_sig  = sigreg_loss(z_reg.reshape(-1, 1024))

            loss = loss_l1 + args.lpips_w * loss_lpips + args.sigreg_w * loss_sig
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

            update_ema(ema_proj, projector, args.ema_decay)
            update_ema(ema_ups,  upsampler,  args.ema_decay)

            epoch_losses.append(loss.item())
            global_step += 1

            if global_step % args.log_every == 0:
                avg = np.mean(epoch_losses[-args.log_every:])
                lr  = optimizer.param_groups[0]["lr"]
                z_mean = z_reg.detach().mean().item()
                z_std  = z_reg.detach().std().item()
                print(f"  ep{epoch+1} step{global_step}  "
                      f"loss={avg:.4e}  l1={loss_l1.item():.4e}  "
                      f"lpips={loss_lpips.item():.4e}  "
                      f"sig={loss_sig.item():.4e}  "
                      f"z(μ={z_mean:.2f} σ={z_std:.2f})  lr={lr:.2e}", flush=True)
                if args.wandb:
                    wb.log({"train/loss": avg, "train/l1": loss_l1.item(),
                            "train/lpips": loss_lpips.item(),
                            "train/sigreg": loss_sig.item(),
                            "train/z_mean": z_mean, "train/z_std": z_std,
                            "train/lr": lr}, step=global_step)

            # ── Validation images ──────────────────────────────────────────────
            if global_step % args.val_every == 0:
                projector.eval(); upsampler.eval(); decoder.eval()
                with torch.no_grad():
                    val_imgs, _ = next(iter(val_loader))
                    val_imgs = val_imgs[:8].to(device)
                    val_norm = (val_imgs - enc_mean) / enc_std
                    val_feats = encoder.model.forward_features(val_norm)
                    val_reg   = val_feats["x_storage_tokens"]
                    val_z     = projector(val_reg)
                    val_up    = upsampler(val_z)
                    val_dec   = decoder(val_up, drop_cls_token=False).logits
                    val_rec   = (decoder.unpatchify(val_dec) * enc_std + enc_mean).clamp(0,1)
                    # side-by-side: original | reconstruction
                    panel = torch.cat([val_imgs, val_rec], dim=0)
                    out_path = os.path.join(args.out_dir, f"val_step{global_step:07d}.png")
                    save_image(panel, out_path, nrow=8, normalize=False)
                    print(f"  Saved {out_path}", flush=True)
                    if args.wandb:
                        wb.log({"val/images": wb.Image(out_path,
                                caption=f"top=original  bottom=recon  step={global_step}")},
                               step=global_step)
                projector.train(); upsampler.train(); decoder.train()

        elapsed = time.time() - t0
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{args.epochs}  avg_loss={avg_loss:.4e}  "
              f"time={elapsed:.0f}s", flush=True)
        if args.wandb:
            wb.log({"epoch/loss": avg_loss}, step=global_step)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (epoch + 1) % args.ckpt_every == 0:
            ckpt = {"epoch": epoch+1, "global_step": global_step,
                    "projector": projector.state_dict(),
                    "upsampler": upsampler.state_dict(),
                    "decoder":   decoder.state_dict(),
                    "ema_proj":  ema_proj.state_dict(),
                    "ema_ups":   ema_ups.state_dict(),
                    "optimizer": optimizer.state_dict()}
            path = os.path.join(args.out_dir, f"ckpt_ep{epoch+1:03d}.pt")
            torch.save(ckpt, path)
            print(f"  Saved {path}", flush=True)

    if args.wandb:
        wb.finish()
    print("Training done.")


if __name__ == "__main__":
    main()
