#!/usr/bin/env python3
"""Unified config-driven stage-1 MLS-decoder trainer.

Replaces the 9 forked train_decoder_mls*.py scripts with one trainer + a YAML
per experiment (configs/stage1/decoder/*.yaml). The only thing that varies
between experiments is the `combine` module (stage1.combine.MLSCombine),
instantiated from config; the recipe (frozen DINOv3 -> combine -> ViT-XL decoder,
L1+LPIPS+GAN, optional SIGReg, EMA, val-1k PSNR/SSIM, LOO/solo probes) is shared.

Usage:
    torchrun --nproc_per_node=4 src/train_decoder.py \
        --config configs/stage1/decoder/random-drop-layer-mls-mlp-sigreg-k23.yaml
"""

import sys, os, math, argparse, time, glob
from copy import deepcopy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from torchvision import transforms

from configs.stage1_decoder import DecoderConfig
from encoders.vision_encoder import create_encoder
from stage1.rae import _load_decoder
from stage1.disc import DinoDiscriminator, hinge_d_loss, vanilla_g_loss, calculate_adaptive_weight
from stage1.disc.diffaug import DiffAug
from stage1.disc.lpips import LPIPS as LPIPS_
from stage1.disc.utils import RandomWindowCrop
from overfit_sigreg import sigreg_loss, gaussian_diag, psnr, val_recon_psnr_npz


def update_ema(ema, model, decay=0.9995):
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), model.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1 - decay)
        for eb, b in zip(ema.buffers(), model.buffers()):    # BN running stats: copy
            eb.copy_(b)


def make_loader(data_dir, image_size, batch_size, num_workers, world_size, rank):
    t = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    arrow_dir = os.path.join(data_dir, "imagenet-latents-images")
    if os.path.isdir(arrow_dir):
        from data.partial_imagenet import PartialImageNetDataset
        ds = PartialImageNetDataset(data_dir, split="train", transform=t)
    else:
        from torchvision.datasets import ImageFolder
        ds = ImageFolder(os.path.join(data_dir, "train"), transform=t)
    sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True), sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="decoder training YAML")
    args = ap.parse_args()
    cfg: DecoderConfig = OmegaConf.to_object(OmegaConf.merge(
        OmegaConf.structured(DecoderConfig), OmegaConf.load(args.config)))
    C, L, D, T, P = cfg.combine, cfg.loss, cfg.data, cfg.training, cfg.probe

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(T.seed + rank)
    torch.backends.cudnn.benchmark = True
    is_main = rank == 0
    if is_main:
        os.makedirs(T.out_dir, exist_ok=True)
        print(f"World {world}  batch/GPU {T.batch_size} (global {T.batch_size*world})")
        print(f"combine: {C.target}  params={C.params}")

    if cfg.wandb.enabled and is_main:
        import wandb
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                   name=cfg.wandb.name, config=OmegaConf.to_container(OmegaConf.structured(cfg)),
                   tags=["decoder", "stage1"])

    # -- data ------------------------------------------------------------------
    train_loader, train_sampler = make_loader(
        D.data_dir, D.image_size, T.batch_size, D.num_workers, world, rank)
    if is_main:
        print(f"Train: {len(train_loader.dataset)} images")

    # -- frozen DINOv3 encoder -------------------------------------------------
    layers = list(C.params["layers"])
    layers_str = ".".join(str(x) for x in layers)
    encoder = create_encoder(f"dinov3mls-vit-l16[layers={layers_str}]",
                             device=device, resolution=D.image_size)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    enc_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    enc_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def encode_layers(imgs01):
        imgs_norm = (imgs01 - enc_mean) / enc_std
        return list(encoder.model.get_intermediate_layers(
            imgs_norm, n=encoder.layer_indices, reshape=False,
            return_class_token=False, norm=True))

    latent_dim = cfg.decoder.latent_dim

    # -- combine (instantiated from config) + decoder (from scratch) -----------
    from utils.model_utils import get_obj_from_str
    combine = get_obj_from_str(C.target)(**C.params).to(device)
    decoder = _load_decoder(cfg.decoder.config_path, hidden_size=latent_dim,
                            patch_size=cfg.decoder.patch_size,
                            num_patches=cfg.decoder.num_patches,
                            pretrained_path=None).to(device)

    combine_has_params = combine.has_params if hasattr(combine, "has_params") \
        else any(True for _ in combine.parameters())
    combine_ddp = DDP(combine, device_ids=[local_rank]) if combine_has_params else combine
    decoder_ddp = DDP(decoder, device_ids=[local_rank])
    ema_combine = deepcopy(combine); ema_combine.requires_grad_(False); ema_combine.eval()
    ema_dec = deepcopy(decoder); ema_dec.requires_grad_(False); ema_dec.eval()

    trainable = list(combine.parameters()) + list(decoder.parameters())
    if is_main:
        print(f"Trainable: {sum(p.numel() for p in trainable)/1e6:.1f}M  "
              f"(combine {sum(p.numel() for p in combine.parameters())/1e6:.2f}M + decoder)")

    optimizer = torch.optim.AdamW(trainable, lr=T.lr, betas=(0.9, 0.95), weight_decay=0.0)
    total_steps = T.epochs * len(train_loader)
    warmup_steps = T.warmup_epochs * len(train_loader)

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        pr = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * pr))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    # -- LPIPS + discriminator -------------------------------------------------
    lpips_all = LPIPS_().to(device).eval()
    for p in lpips_all.parameters():
        p.requires_grad_(False)
    disc = DinoDiscriminator(device=device, dino_ckpt_path=L.gan.disc_ckpt,
                             ks=1, recipe="S_8", norm_type="bn", using_spec_norm=True).to(device)
    disc.dino_proxy[0].crop = RandomWindowCrop(D.image_size, 224, 9, False)
    disc.dino_proxy[0].original_input_size = D.image_size
    disc_ddp = DDP(disc, device_ids=[local_rank])
    disc_aug = DiffAug(prob=0.5, cutout=True)
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.9))
    disc_start_step = L.gan.disc_start * len(train_loader)

    # -- pre-encode fixed demo val images (progress signal) --------------------
    val_layers_fixed = val_img_orig = None
    if P.val_image and is_main:
        from PIL import Image as PILImage
        _t = transforms.Compose([transforms.Resize(D.image_size + 32),
                                 transforms.CenterCrop(D.image_size), transforms.ToTensor()])
        vdir = os.path.dirname(os.path.abspath(P.val_image))
        vpaths = sorted(p for p in (glob.glob(os.path.join(vdir, "*.png")) +
                                    glob.glob(os.path.join(vdir, "*.jpg")))
                        if "concat" not in os.path.basename(p).lower()) or [P.val_image]
        val_img_orig = torch.stack([_t(PILImage.open(p).convert("RGB")) for p in vpaths]).to(device)
        with torch.no_grad():
            val_layers_fixed = [t.detach() for t in encode_layers(val_img_orig)]
        print(f"Val demo images: {len(vpaths)} from {vdir}", flush=True)

    # -- auto-resume -----------------------------------------------------------
    start_epoch = global_step = 0
    latest = os.path.join(T.out_dir, "ckpt_latest.pt")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        combine.load_state_dict(ck["combine"]); decoder.load_state_dict(ck["decoder"])
        ema_combine.load_state_dict(ck["ema_combine"]); ema_dec.load_state_dict(ck["ema_dec"])
        disc.load_state_dict(ck["disc"]); optimizer.load_state_dict(ck["optimizer"])
        disc_optimizer.load_state_dict(ck["disc_optimizer"])
        start_epoch, global_step = ck["epoch"], ck["global_step"]
        if "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        else:
            scheduler.last_epoch = global_step - 1; scheduler.step()
        if is_main:
            print(f"Resumed from {latest} (epoch={start_epoch}, step={global_step})")
        del ck

    use_sig = L.sigreg is not None and combine_has_params
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(T.precision == "bf16"))

    def decode_imgs(dec, z):
        out = dec(z, drop_cls_token=False).logits
        m = dec.module if isinstance(dec, DDP) else dec
        return (m.unpatchify(out) * enc_std + enc_mean).clamp(0, 1)

    # -- training loop ---------------------------------------------------------
    for epoch in range(start_epoch, T.epochs):
        train_sampler.set_epoch(epoch)
        if combine_has_params:
            combine_ddp.train()
        decoder_ddp.train()
        t0 = time.time()
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            with torch.no_grad():
                layer_tokens = encode_layers(imgs)
            use_gan = (global_step >= disc_start_step) and (L.gan.disc_weight > 0)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                z = combine_ddp(layer_tokens)
                x_rec = decode_imgs(decoder_ddp, z)
                loss_l1 = F.l1_loss(x_rec, imgs)
                loss_lpips = lpips_all(x_rec * 2 - 1, imgs * 2 - 1).mean()
            loss_rec = loss_l1 + L.lpips_w * loss_lpips
            loss_sig = (sigreg_loss(z.float().reshape(-1, latent_dim),
                                    distributed=L.sigreg.distributed,
                                    scale_by_n=L.sigreg.scale_by_n)
                        if use_sig else torch.zeros((), device=device))

            loss_gan = torch.zeros(1, device=device)
            if use_gan:
                disc_ddp.eval()
                half = max(1, x_rec.shape[0] // 2)
                with autocast_ctx:
                    fake_aug = disc_aug.aug(x_rec[:half] * 2 - 1)
                    logits_fake, _ = disc_ddp(fake_aug, None)
                loss_gan = vanilla_g_loss(logits_fake)
                last_layer = next(reversed(list(decoder_ddp.module.parameters())))
                adp_w = calculate_adaptive_weight(loss_rec, loss_gan, last_layer).clamp(0, 1e4).detach()
            else:
                adp_w = torch.tensor(0.0, device=device)

            sig_w = L.sigreg.weight if use_sig else 0.0
            loss = loss_rec + sig_w * loss_sig + L.gan.disc_weight * adp_w * loss_gan
            loss.backward()
            if T.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(trainable, T.clip_grad)
            optimizer.step(); scheduler.step()

            if use_gan:
                disc_ddp.train()
                disc_optimizer.zero_grad(set_to_none=True)
                with autocast_ctx:
                    real_aug = disc_aug.aug(imgs[:half] * 2 - 1)
                    fake_aug = disc_aug.aug(x_rec[:half].detach() * 2 - 1)
                    logits_real, _ = disc_ddp(real_aug, None)
                    logits_fake, _ = disc_ddp(fake_aug, None)
                    loss_d = hinge_d_loss(logits_real, logits_fake)
                loss_d.backward(); disc_optimizer.step()

            update_ema(ema_combine, combine, T.ema_decay)
            update_ema(ema_dec, decoder, T.ema_decay)
            global_step += 1

            # -- step-interval val: PSNR/SSIM on a FIXED random val-N subset ----
            # (seed 0 -> identical images every step and across runs -> the curve
            #  reflects model improvement, not sampling noise). RANK 0 ONLY, and
            #  NO dist.barrier(): the eval is collective-free (frozen encoder +
            #  non-DDP EMA modules), so rank 0 just falls ~Ns behind and rejoins
            #  at the next training collective. A barrier here deadlocks against
            #  the next step's SIGReg broadcast (NCCL op-mismatch).
            if (T.val_every_steps > 0 and global_step % T.val_every_steps == 0
                    and is_main and D.val_npz and os.path.exists(D.val_npz)):
                def _recon(imgs01):
                    return decode_imgs(ema_dec, ema_combine(encode_layers(imgs01)))
                vpsnr, vssim = val_recon_psnr_npz(_recon, D.val_npz, device,
                                                  n=D.val_n, seed=0, want_ssim=True)
                tsv = os.path.join(T.out_dir, "val_psnr_steps.tsv")
                if not os.path.exists(tsv):
                    with open(tsv, "w") as f:
                        f.write("step\tpsnr\tssim\n")
                with open(tsv, "a") as f:
                    f.write(f"{global_step}\t{vpsnr:.4f}\t{vssim:.4f}\n")
                print(f"  [val@{global_step}] PSNR={vpsnr:.3f} dB  SSIM={vssim:.4f}  "
                      f"({D.val_n} val imgs)", flush=True)
                if cfg.wandb.enabled:
                    import wandb
                    wandb.log({"val/psnr": vpsnr, "val/ssim": vssim}, step=global_step)

            if is_main and global_step % T.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                zd = gaussian_diag(z); ps = psnr(x_rec, imgs)
                print(f"  ep{epoch+1} s{global_step}  loss={loss.item():.4e}  l1={loss_l1.item():.4e}"
                      f"  lpips={loss_lpips.item():.4e}  psnr={ps:.2f}  sig={loss_sig.item():.4e}"
                      f"  gan={loss_gan.item():.4e}  z(mu={zd['mean']:.2f} sd={zd['std']:.2f})"
                      f"  vard(mu={zd['var_mean']:.2f})  lr={lr:.2e}", flush=True)
                if cfg.wandb.enabled:
                    import wandb
                    wandb.log({"train/loss": loss.item(), "train/l1": loss_l1.item(),
                               "train/lpips": loss_lpips.item(), "train/psnr": ps,
                               "train/sigreg": loss_sig.item(), "train/gan_g": loss_gan.item(),
                               "train/z_mean": zd["mean"], "train/z_std": zd["std"],
                               "train/z_var_mean": zd["var_mean"], "train/lr": lr}, step=global_step)

        if is_main:
            print(f"Epoch {epoch+1}/{T.epochs}  time={time.time()-t0:.0f}s", flush=True)

        # -- per-epoch val ------------------------------------------------------
        if is_main and val_layers_fixed is not None:
            with torch.no_grad():
                vrec = decode_imgs(ema_dec, ema_combine(val_layers_fixed))
            vps_demo = psnr(vrec, val_img_orig)
            print(f"  Val PSNR (EMA, {vrec.shape[0]} demo imgs): {vps_demo:.2f} dB", flush=True)
            if cfg.wandb.enabled:
                import wandb; wandb.log({"val/psnr_demo": vps_demo}, step=global_step)

        if is_main and D.val_npz and os.path.exists(D.val_npz):
            def _recon(imgs01):
                return decode_imgs(ema_dec, ema_combine(encode_layers(imgs01)))
            vpsnr, vssim = val_recon_psnr_npz(_recon, D.val_npz, device, n=D.val_n,
                                              seed=0, want_ssim=True)
            print(f"  Val PSNR (EMA, {D.val_n} val imgs): {vpsnr:.3f} dB  SSIM={vssim:.4f}", flush=True)
            if cfg.wandb.enabled:
                import wandb; wandb.log({"val/psnr": vpsnr, "val/ssim": vssim}, step=global_step)

        # -- LOO/solo probes ----------------------------------------------------
        do_probe = (P.loo_solo == "every") or (P.loo_solo == "final" and epoch + 1 == T.epochs)
        if is_main and do_probe and val_layers_fixed is not None:
            with torch.no_grad():
                def _sub_psnr(idx):
                    return psnr(decode_imgs(ema_dec, ema_combine(val_layers_fixed, idx=idx)), val_img_orig)
                full = psnr(decode_imgs(ema_dec, ema_combine(val_layers_fixed)), val_img_orig)
                K = len(val_layers_fixed)
                loo = [_sub_psnr([j for j in range(K) if j != i]) for i in range(K)]
                solo = [_sub_psnr([i]) for i in range(K)]
            dloo = [full - v for v in loo]
            print("  Val LOO dPSNR = [" + " ".join(f"{v:+.2f}" for v in dloo) + f"]  layers={layers}", flush=True)
            print("  Val solo PSNR = [" + " ".join(f"{v:.2f}" for v in solo) + "]", flush=True)
            if cfg.wandb.enabled:
                import wandb
                wandb.log({**{f"val/loo_dpsnr_L{l}": v for l, v in zip(layers, dloo)},
                           **{f"val/solo_psnr_L{l}": v for l, v in zip(layers, solo)}}, step=global_step)

        # -- checkpoint ---------------------------------------------------------
        if is_main:
            ckpt = {"epoch": epoch + 1, "global_step": global_step, "layers": layers,
                    "combine": combine.state_dict(), "decoder": decoder.state_dict(),
                    "ema_combine": ema_combine.state_dict(), "ema_dec": ema_dec.state_dict(),
                    "disc": disc.state_dict(), "optimizer": optimizer.state_dict(),
                    "disc_optimizer": disc_optimizer.state_dict(), "scheduler": scheduler.state_dict()}
            torch.save(ckpt, latest)
            if (epoch + 1) % T.ckpt_every == 0:
                torch.save(ckpt, os.path.join(T.out_dir, f"ckpt_ep{epoch+1:03d}.pt"))
            print(f"  Saved ckpt (ep{epoch+1})", flush=True)

    if cfg.wandb.enabled and is_main:
        import wandb; wandb.finish()
    dist.destroy_process_group()
    if is_main:
        print("Done.")


if __name__ == "__main__":
    main()
