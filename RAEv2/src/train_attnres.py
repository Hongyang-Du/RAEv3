#!/usr/bin/env python3
"""
Multi-GPU DDP training: SpatialAttnRes + SIGReg + ViT Decoder on ImageNet.

Architecture:
  DINOv3-L (frozen, layers [11,13,15,17,19,21,23])
       ↓ patch tokens [B, 7, 256, 1024]
  SpatialAttnRes (learned: per-position AttnRes + gated FFN)
       ↓ z [B, 256, 1024]   ← SIGReg (all_gather for multi-GPU)
  ViT Decoder (fine-tuned from pretrained)
       ↓ x_rec [B, 3, 256, 256]

Loss: L1 + LPIPS + λ_sig * SIGReg

Usage:
    torchrun --nproc_per_node=8 src/train_attnres.py \
        --data /home/colligo/data/imagenet-256 \
        --epochs 20 --batch-size 128 --wandb
"""

import sys, os, math, argparse, time, glob
from copy import deepcopy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image

from models.spatial_attn_res import SpatialAttnRes, RAEV2_LAYERS
from overfit_sigreg import sigreg_loss, load_lpips
from stage1.disc import DinoDiscriminator, hinge_d_loss, vanilla_g_loss, calculate_adaptive_weight
from stage1.disc.diffaug import DiffAug


# ─── EMA ─────────────────────────────────────────────────────────────────────

def update_ema(ema, model, decay=0.9995):
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), model.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1 - decay)


# ─── Data ────────────────────────────────────────────────────────────────────

def make_loader(data_dir, split, image_size, batch_size,
                num_workers=4, world_size=1, rank=0):
    """
    Supports:
      1. HuggingFace Arrow (RAEv2 format): data_dir contains imagenet-latents-images/
      2. ImageFolder (fallback):           data_dir/train/<class>/*.jpg
    """
    t_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    t_val = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    t = t_train if split == "train" else t_val

    arrow_dir = os.path.join(data_dir, "imagenet-latents-images")
    if os.path.isdir(arrow_dir):
        # Partial or full HuggingFace Arrow download
        from data.partial_imagenet import PartialImageNetDataset
        ds = PartialImageNetDataset(data_dir, split=split, transform=t)
    else:
        # Standard ImageFolder (data_dir/train/<class>/...)
        from torchvision.datasets import ImageFolder
        ds = ImageFolder(os.path.join(data_dir, split), transform=t)

    sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=world_size, rank=rank,
        shuffle=(split == "train"), drop_last=True,
    )
    return (
        torch.utils.data.DataLoader(
            ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        ),
        sampler,
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--data",         required=True)
    parser.add_argument("--image-size",   type=int, default=256)
    # training
    parser.add_argument("--epochs",       type=int, default=20)
    parser.add_argument("--batch-size",   type=int, default=128,
                        help="Per-GPU batch size")
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--warmup-epochs",type=int, default=2)
    parser.add_argument("--sigreg-w",     type=float, default=0.1)
    parser.add_argument("--lpips-w",      type=float, default=1.0)
    # GAN
    parser.add_argument("--disc-weight",  type=float, default=0.75)
    parser.add_argument("--disc-start",   type=int,   default=1,
                        help="Epoch to start GAN training")
    parser.add_argument("--disc-ckpt",    type=str,
                        default="pretrained_models/encoders/dino/dino_vit_small_patch8_224.pth")
    parser.add_argument("--ema-decay",    type=float, default=0.9995)
    parser.add_argument("--val-image",    type=str,   default=None,
                        help="Fixed image to reconstruct every epoch for visual tracking")
    parser.add_argument("--clip-grad",    type=float, default=1.0)
    # model
    parser.add_argument("--layers",       type=int, nargs="+",
                        default=RAEV2_LAYERS,
                        help="DINOv3 layer indices (default: RAEv2 K7)")
    parser.add_argument("--latent-dim",   type=int, default=1024)
    # logging
    parser.add_argument("--log-every",    type=int, default=50)
    parser.add_argument("--val-every",    type=int, default=500)
    parser.add_argument("--ckpt-every",   type=int, default=2)
    parser.add_argument("--out-dir",      default="output/train_attnres")
    parser.add_argument("--wandb",        action="store_true")
    parser.add_argument("--wandb-project",default="raev3")
    parser.add_argument("--wandb-entity", default="hongyangd")
    parser.add_argument("--num-workers",  type=int, default=4)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--precision",    type=str, default="fp32",
                        choices=["fp32", "bf16"],
                        help="Training precision. bf16 halves memory usage.")
    args = parser.parse_args()

    # ── DDP init ──────────────────────────────────────────────────────────────
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    is_main = (rank == 0)

    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"World size: {world_size}  |  batch/GPU: {args.batch_size}"
              f"  |  global batch: {args.batch_size * world_size}")
        print(f"Layers: {args.layers}")

    # ── wandb ─────────────────────────────────────────────────────────────────
    if args.wandb and is_main:
        import wandb
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"attnres-k{len(args.layers)}-sigreg",
            config={**vars(args), "global_batch": args.batch_size * world_size},
            tags=["attnres", "sigreg", "stage1", f"k{len(args.layers)}"],
        )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, train_sampler = make_loader(
        args.data, "train", args.image_size, args.batch_size,
        args.num_workers, world_size, rank,
    )
    if is_main:
        print(f"Train: {len(train_loader.dataset)} images")

    # ── DINOv3 encoder (frozen) ───────────────────────────────────────────────
    from encoders.vision_encoder import create_encoder
    encoder = create_encoder("dinov3-vit-l16", device=device,
                             resolution=args.image_size)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    enc_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    enc_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    # ── Pre-encode fixed val image (encoder frozen, do once) ─────────────────
    val_patches_fixed = None
    val_img_orig      = None
    if args.val_image and is_main and os.path.exists(args.val_image):
        from torchvision import transforms
        from PIL import Image as PILImage
        _t = transforms.Compose([
            transforms.Resize(args.image_size + 32),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ])
        val_img_orig = _t(PILImage.open(args.val_image).convert("RGB")
                         ).unsqueeze(0).to(device)
        with torch.no_grad():
            _norm = (val_img_orig - enc_mean) / enc_std
            val_patches_fixed = encoder.model.get_intermediate_layers(
                _norm, n=args.layers, return_class_token=False
            )
            val_patches_fixed = [p.detach() for p in val_patches_fixed]
        if is_main:
            print(f"Val image pre-encoded: {args.val_image}", flush=True)

    # ── Decoder (fine-tune from pretrained) ───────────────────────────────────
    from omegaconf import OmegaConf
    from stage1.rae import _load_decoder
    s1 = OmegaConf.load(
        "configs/stage2/training/imagenet-dinov3l-k7.yaml"
    ).stage_1.params
    decoder = _load_decoder(
        s1.decoder_config_path,
        hidden_size=args.latent_dim,
        patch_size=16,
        num_patches=256,
        pretrained_path=s1.pretrained_decoder_path,
    ).to(device)

    # ── SpatialAttnRes (learned) ───────────────────────────────────────────────
    attn_res = SpatialAttnRes(
        dim=1024, out_dim=args.latent_dim, gate_init=-5.0
    ).to(device)

    # DDP wrap
    attn_res_ddp = DDP(attn_res, device_ids=[local_rank])
    decoder_ddp  = DDP(decoder,  device_ids=[local_rank])

    # EMA
    ema_attn = deepcopy(attn_res); ema_attn.requires_grad_(False)
    ema_dec  = deepcopy(decoder);  ema_dec.requires_grad_(False)

    trainable = (list(attn_res.parameters()) + list(decoder.parameters()))
    if is_main:
        n = sum(p.numel() for p in trainable) / 1e6
        print(f"Trainable: {n:.1f}M  (SpatialAttnRes + decoder)")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0
    )
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    # ── LPIPS ─────────────────────────────────────────────────────────────────
    from stage1.disc.lpips import LPIPS as LPIPS_
    lpips_all = LPIPS_().to(device).eval()
    for p in lpips_all.parameters():
        p.requires_grad_(False)

    # ── Discriminator ─────────────────────────────────────────────────────────
    disc = DinoDiscriminator(
        device=device,
        dino_ckpt_path=args.disc_ckpt,
        ks=1,
        recipe="S_8",
        norm_type="bn",
        using_spec_norm=True,
    ).to(device)
    # Patch crop to handle 256×256 input (disc was designed for 224)
    from stage1.disc.utils import RandomWindowCrop
    disc.dino_proxy[0].crop = RandomWindowCrop(args.image_size, 224, 9, False)
    disc.dino_proxy[0].original_input_size = args.image_size
    disc_ddp = DDP(disc, device_ids=[local_rank])
    disc_aug = DiffAug(prob=0.5, cutout=True)
    disc_optimizer = torch.optim.Adam(
        disc.parameters(), lr=1e-4, betas=(0.5, 0.9)
    )
    disc_start_step = args.disc_start * len(train_loader)
    if is_main:
        n_disc = sum(p.numel() for p in disc.parameters()) / 1e6
        print(f"Discriminator: {n_disc:.1f}M  GAN starts at epoch {args.disc_start}")

    # ── Auto-resume from latest checkpoint ────────────────────────────────────
    start_epoch = 0
    global_step = 0
    latest = os.path.join(args.out_dir, "ckpt_latest.pt")
    if os.path.exists(latest):
        ckpt = torch.load(latest, map_location="cpu")
        attn_res.load_state_dict(ckpt["attn_res"])
        decoder.load_state_dict(ckpt["decoder"])
        ema_attn.load_state_dict(ckpt["ema_attn"])
        ema_dec.load_state_dict(ckpt["ema_dec"])
        disc.load_state_dict(ckpt["disc"])
        optimizer.load_state_dict(ckpt["optimizer"])
        disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
        start_epoch  = ckpt["epoch"]
        global_step  = ckpt["global_step"]
        if is_main:
            print(f"Resumed from {latest}  (epoch={start_epoch}, step={global_step})")

    # ── Precision ─────────────────────────────────────────────────────────────
    use_bf16 = (args.precision == "bf16")
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)
    if is_main:
        print(f"Precision: {args.precision}")

    # ── Training ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        attn_res_ddp.train(); decoder_ddp.train()
        t0 = time.time()

        for step, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)

            # DINOv3 encode (frozen) — get patch tokens from K layers
            with torch.no_grad():
                imgs_norm = (imgs - enc_mean) / enc_std
                layer_out = encoder.model.get_intermediate_layers(
                    imgs_norm, n=args.layers, return_class_token=False
                )
                patches_list = [p.detach() for p in layer_out]
                # each: [B, 256, 1024]

            use_gan = (global_step >= disc_start_step) and (args.disc_weight > 0)

            # ── Step A: Generator (attn_res + decoder) ─────────────────────────
            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx:
                z = attn_res_ddp(patches_list)                    # [B, 256, 1024]
                dec_out = decoder_ddp(z, drop_cls_token=False).logits
                x_rec   = (decoder_ddp.module.unpatchify(dec_out)
                           * enc_std + enc_mean).clamp(0, 1)

                loss_l1    = F.l1_loss(x_rec, imgs)
                loss_lpips = lpips_all(x_rec * 2 - 1, imgs * 2 - 1).mean()
            loss_sig   = sigreg_loss(z.float().reshape(-1, args.latent_dim))
            loss_rec   = loss_l1 + args.lpips_w * loss_lpips

            loss_gan = torch.zeros(1, device=device)
            if use_gan:
                torch.cuda.empty_cache()
                disc_ddp.eval()
                half = max(1, x_rec.shape[0] // 2)
                with autocast_ctx:
                    # Generator step: NO detach — need grad for adaptive weight
                    fake_aug = disc_aug.aug(x_rec[:half] * 2 - 1)
                    logits_fake, _ = disc_ddp(fake_aug, None)
                loss_gan = vanilla_g_loss(logits_fake)
                # last trainable decoder parameter as reference for adaptive weight
                last_layer = next(reversed(list(decoder_ddp.module.parameters())))
                adp_w = calculate_adaptive_weight(loss_rec, loss_gan, last_layer)
                adp_w = adp_w.clamp(0, 1e4).detach()
            else:
                adp_w = torch.tensor(0.0, device=device)

            loss = (loss_rec
                    + args.sigreg_w * loss_sig
                    + args.disc_weight * adp_w * loss_gan)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)
            optimizer.step()
            scheduler.step()

            # ── Step B: Discriminator ───────────────────────────────────────────
            if use_gan:
                disc_ddp.train()
                disc_optimizer.zero_grad(set_to_none=True)
                with autocast_ctx:
                    real_aug = disc_aug.aug(imgs[:half]           * 2 - 1)
                    fake_aug = disc_aug.aug(x_rec[:half].detach() * 2 - 1)
                    logits_real, _ = disc_ddp(real_aug, None)
                    logits_fake, _ = disc_ddp(fake_aug, None)
                    loss_d = hinge_d_loss(logits_real, logits_fake)
                loss_d.backward()
                disc_optimizer.step()

            update_ema(ema_attn, attn_res, args.ema_decay)
            update_ema(ema_dec,  decoder,  args.ema_decay)

            global_step += 1

            # ── logging ───────────────────────────────────────────────────────
            if is_main and global_step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                z_m = z.detach().mean().item()
                z_s = z.detach().std().item()
                gate_val = torch.sigmoid(attn_res.gate).mean().item()
                print(f"  ep{epoch+1} s{global_step}"
                      f"  loss={loss.item():.4e}"
                      f"  l1={loss_l1.item():.4e}"
                      f"  lpips={loss_lpips.item():.4e}"
                      f"  sig={loss_sig.item():.4e}"
                      f"  gan={loss_gan.item():.4e}"
                      f"  gate={gate_val:.3f}"
                      f"  z(μ={z_m:.2f} σ={z_s:.2f})"
                      f"  lr={lr:.2e}", flush=True)
                if args.wandb:
                    import wandb
                    wandb.log({
                        "train/loss":   loss.item(),
                        "train/l1":     loss_l1.item(),
                        "train/lpips":  loss_lpips.item(),
                        "train/sigreg": loss_sig.item(),
                        "train/gan_g":  loss_gan.item(),
                        "train/adp_w":  adp_w.item(),
                        "train/gate":   gate_val,
                        "train/z_mean": z_m, "train/z_std": z_s,
                        "train/lr":     lr,
                    }, step=global_step)


        if is_main:
            print(f"Epoch {epoch+1}/{args.epochs}  "
                  f"time={time.time()-t0:.0f}s", flush=True)

        # ── fixed val image reconstruction (one per epoch) ───────────────────
        if is_main and val_patches_fixed is not None:
            attn_res_ddp.eval(); decoder_ddp.eval()
            with torch.no_grad():
                val_z   = ema_attn(val_patches_fixed)
                val_dec = ema_dec(val_z, drop_cls_token=False).logits
                val_rec = (ema_dec.unpatchify(val_dec)
                           * enc_std + enc_mean).clamp(0, 1)
            panel = torch.cat([val_img_orig, val_rec], dim=-1)  # side by side
            out_path = os.path.join(args.out_dir, f"val_ep{epoch+1:03d}.png")
            save_image(panel.squeeze(0), out_path)
            print(f"  Val image → {out_path}", flush=True)
            if args.wandb:
                import wandb
                wandb.log({"val/recon": wandb.Image(
                    out_path, caption=f"left=orig  right=recon  ep={epoch+1}"
                )}, step=global_step)
            attn_res_ddp.train(); decoder_ddp.train()

        # ── checkpoint ────────────────────────────────────────────────────────
        if is_main and (epoch + 1) % args.ckpt_every == 0:
            ckpt = {
                "epoch":        epoch + 1,
                "global_step":  global_step,
                "attn_res":     attn_res.state_dict(),
                "decoder":      decoder.state_dict(),
                "ema_attn":     ema_attn.state_dict(),
                "ema_dec":      ema_dec.state_dict(),
                "disc":         disc.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "disc_optimizer": disc_optimizer.state_dict(),
                "layers":       args.layers,
            }
            import shutil
            # save directly to ckpt_latest.pt — no epoch file kept on disk
            latest = os.path.join(args.out_dir, "ckpt_latest.pt")
            torch.save(ckpt, latest)
            print(f"  Saved {latest}", flush=True)

    if args.wandb and is_main:
        import wandb; wandb.finish()
    dist.destroy_process_group()
    if is_main:
        print("Done.")


if __name__ == "__main__":
    main()
