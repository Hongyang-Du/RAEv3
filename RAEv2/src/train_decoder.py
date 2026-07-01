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

import sys, os, math, argparse, time, glob, threading
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

    # Node-count-agnostic overrides: fix per-GPU batch (OOM-safe) and let the launch
    # script scale lr/epochs/warmup/disc_start with the actual global batch (= per-GPU
    # batch x world_size), so the run trains ~the same total passes on any node count.
    if os.environ.get("BATCH_SIZE_OVERRIDE"): T.batch_size = int(os.environ["BATCH_SIZE_OVERRIDE"])
    if os.environ.get("LR_OVERRIDE"): T.lr = float(os.environ["LR_OVERRIDE"])
    if os.environ.get("EPOCHS_OVERRIDE"): T.epochs = int(os.environ["EPOCHS_OVERRIDE"])
    if os.environ.get("WARMUP_OVERRIDE"): T.warmup_epochs = int(os.environ["WARMUP_OVERRIDE"])
    if os.environ.get("DISC_START_OVERRIDE"): L.gan.disc_start = int(os.environ["DISC_START_OVERRIDE"])
    if os.environ.get("OUT_DIR_OVERRIDE"): T.out_dir = os.environ["OUT_DIR_OVERRIDE"]
    if os.environ.get("NUM_WORKERS_OVERRIDE"): D.num_workers = int(os.environ["NUM_WORKERS_OVERRIDE"])
    # preemption resilience: also save ckpt_latest every N steps (not just per epoch), so
    # a preempted job resumes from a recent step instead of restarting the epoch from zero.
    ckpt_every_steps = int(os.environ.get("CKPT_EVERY_STEPS", "0") or "0")
    no_ema = bool(int(os.environ.get("NO_EMA", "0") or "0"))   # skip saving ema_* copies (raw==ema at convergence); resume re-inits ema from raw

    from datetime import timedelta
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))   # NFS ckpt saves can be slow
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
        # val runs async (background thread, after training has advanced past this
        # step), so give val/* its own x-axis -> wandb won't drop it for a step regression
        wandb.define_metric("val/step")
        wandb.define_metric("val/*", step_metric="val/step")

    # -- data ------------------------------------------------------------------
    if getattr(D, "mix", None):
        # Multi-source weighted mix (official general recipe): imagenet (local arrow)
        # + blip3o/rendertext/flux streamed from HF (cached to localssd). Reuses the
        # stage-1 MixedDataloader; it yields (image, cond) and acts as its own loader
        # + sampler (set_epoch / __len__), so the loop below works unchanged.
        from data import prepare_unified_dataloader
        mix_cfg = {
            "mix": OmegaConf.to_container(D.mix, resolve=True) if OmegaConf.is_config(D.mix) else D.mix,
            "seed": T.seed,
        }
        mixed = prepare_unified_dataloader(
            config=mix_cfg, image_size=D.image_size, batch_size=T.batch_size,
            num_workers=D.num_workers, rank=rank, world_size=world,
            transform=None, condition_type="label", shuffle=True,
            virtual_epoch_steps=getattr(D, "virtual_epoch_steps", None),
        )
        train_loader = train_sampler = mixed   # MixedDataloader: loader + set_epoch + __len__
        if is_main:
            print(f"Train mix: {len(mixed)} steps/epoch over {mixed.dataset_size:,} weighted samples "
                  f"({mixed.child_names})")
    else:
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

    # -- pre-load the fixed val-N subset ONCE (rank 0) -------------------------
    # The 9.8GB val npz is loaded/decoded entirely on first access; doing that
    # at an epoch boundary (between train_one_epoch and the next epoch's first
    # SIGReg broadcast) makes the other ranks time out waiting. Extract the 100
    # fixed images here, before the loop, into a small CPU tensor (~20MB).
    val_eval_imgs = None
    if is_main and D.val_npz and os.path.exists(D.val_npz):
        import numpy as np
        z = np.load(D.val_npz, mmap_mode="r")
        arr = z[z.files[0]] if hasattr(z, "files") else z
        idx = torch.randperm(len(arr), generator=torch.Generator().manual_seed(0))[:D.val_n].tolist()
        val_eval_imgs = torch.stack([torch.from_numpy(arr[j].copy()) for j in idx]) \
            .permute(0, 3, 1, 2).contiguous()        # [N,3,256,256] uint8, CPU
        del z, arr
        print(f"Val-set images pre-loaded: {val_eval_imgs.shape[0]} from {D.val_npz}", flush=True)

    # -- auto-resume -----------------------------------------------------------
    start_epoch = global_step = 0
    latest = os.path.join(T.out_dir, "ckpt_latest.pt")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        combine.load_state_dict(ck["combine"]); decoder.load_state_dict(ck["decoder"])
        ema_combine.load_state_dict(ck.get("ema_combine") or ck["combine"]); ema_dec.load_state_dict(ck.get("ema_dec") or ck["decoder"])  # NO_EMA: re-init ema from raw
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
    elif getattr(T, "init_from", None):
        # warm-start: load weights from an external ckpt but train fresh (epoch 0,
        # new optimizer/scheduler/LR). Used to fine-tune a drop-trained model with
        # full layers (p_drop=0) to close the train/inference gap.
        ck = torch.load(T.init_from, map_location="cpu", weights_only=False)
        combine.load_state_dict(ck["combine"]); decoder.load_state_dict(ck["decoder"])
        ema_combine.load_state_dict(ck.get("ema_combine") or ck["combine"]); ema_dec.load_state_dict(ck.get("ema_dec") or ck["decoder"])  # NO_EMA: re-init ema from raw
        if "disc" in ck:
            disc.load_state_dict(ck["disc"])
        if is_main:
            print(f"Warm-started weights from {T.init_from} (fresh optimizer, epoch=0)")
        del ck

    # SIGReg regularizes the projector's latent -> it only has effect with a
    # parametric projector, and the proven recipe (LeWM) is the BN-MLP. Forbid the
    # silent no-op (projector=none) and the untested ln combo: sigreg => bn.
    if L.sigreg is not None and C.params.get("projector") != "bn":
        raise ValueError(
            f"loss.sigreg is set but combine.projector={C.params.get('projector')!r}; "
            "SIGReg requires projector: bn (BN-MLP). Set projector: bn or disable sigreg.")
    use_sig = L.sigreg is not None and combine_has_params
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(T.precision == "bf16"))

    def decode_imgs(dec, z):
        out = dec(z, drop_cls_token=False).logits
        m = dec.module if isinstance(dec, DDP) else dec
        # decoder predicts DIRECTLY in [0,1] pixel space (official RAE convention:
        # RAE.decode returns unpatchify(out) raw). No ImageNet de-norm -> the trained
        # decoder is a drop-in for stage1.RAE (offline_eval_stage1) exactly like the
        # official k7/k23 decoders. (Old convention was normalized-output * std + mean.)
        return m.unpatchify(out).clamp(0, 1)

    # -- async val/ckpt: rank 0 snapshots (frozen EMA copy + CPU-cloned ckpt),
    # then a daemon thread runs val + torch.save while the next epoch trains.
    # The snapshot makes it race-free; no collective inside -> no deadlock.
    def _to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.detach().to("cpu", copy=True)
        if isinstance(obj, dict):
            return {k: _to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_cpu(v) for v in obj]
        return obj

    def _async_val_save(sn_combine, sn_dec, ckpt_cpu, ep, gstep, do_probe):
        wb = None
        if cfg.wandb.enabled:
            import wandb as wb
        # -- demo val
        if val_layers_fixed is not None:
            with torch.no_grad():
                vrec = decode_imgs(sn_dec, sn_combine(val_layers_fixed))
            vps_demo = psnr(vrec, val_img_orig)
            print(f"  [val ep{ep}] PSNR (EMA, {vrec.shape[0]} demo imgs): {vps_demo:.2f} dB", flush=True)
            if wb is not None:
                wb.log({"val/psnr_demo": vps_demo, "val/step": gstep})
        # -- val-N subset (functional SSIM: no Metric.compute() PG sync)
        if val_eval_imgs is not None:
            from torchmetrics.functional.image import structural_similarity_index_measure as _ssim_fn
            psnrs, ssims = [], []
            with torch.no_grad():
                for i in range(0, val_eval_imgs.shape[0], 32):
                    vb = val_eval_imgs[i:i + 32].to(device).float() / 255
                    rec = decode_imgs(sn_dec, sn_combine(encode_layers(vb))).clamp(0, 1).float()
                    mse = ((rec - vb) ** 2).flatten(1).mean(1)
                    psnrs.append(-10.0 * torch.log10(mse + 1e-10))
                    ssims.append(_ssim_fn(rec, vb, data_range=1.0).item() * rec.shape[0])
            vpsnr = torch.cat(psnrs).mean().item()
            vssim = sum(ssims) / val_eval_imgs.shape[0]
            print(f"  [val ep{ep}] PSNR (EMA, {D.val_n} val imgs): {vpsnr:.3f} dB  SSIM={vssim:.4f}", flush=True)
            tsv = os.path.join(T.out_dir, "val_psnr_steps.tsv")
            if not os.path.exists(tsv):
                with open(tsv, "w") as f:
                    f.write("step\tpsnr\tssim\n")
            with open(tsv, "a") as f:
                f.write(f"{gstep}\t{vpsnr:.4f}\t{vssim:.4f}\n")
            if wb is not None:
                wb.log({"val/psnr": vpsnr, "val/ssim": vssim, "val/step": gstep})
        # -- LOO/solo probes (final epoch only by default)
        if do_probe and val_layers_fixed is not None:
            with torch.no_grad():
                def _sub_psnr(idx):
                    return psnr(decode_imgs(sn_dec, sn_combine(val_layers_fixed, idx=idx)), val_img_orig)
                full = psnr(decode_imgs(sn_dec, sn_combine(val_layers_fixed)), val_img_orig)
                K = len(val_layers_fixed)
                loo = [_sub_psnr([j for j in range(K) if j != i]) for i in range(K)]
                solo = [_sub_psnr([i]) for i in range(K)]
            dloo = [full - v for v in loo]
            print("  [val ep%d] LOO dPSNR = [" % ep + " ".join(f"{v:+.2f}" for v in dloo) + f"]  layers={layers}", flush=True)
            print("  [val ep%d] solo PSNR = [" % ep + " ".join(f"{v:.2f}" for v in solo) + "]", flush=True)
            if wb is not None:
                wb.log({**{f"val/loo_dpsnr_L{l}": v for l, v in zip(layers, dloo)},
                        **{f"val/solo_psnr_L{l}": v for l, v in zip(layers, solo)}, "val/step": gstep})
        # -- checkpoint (CPU-cloned, safe to save while training mutates the live model)
        # atomic write: tmp + rename so a preemption/quota-exceeded mid-save never leaves
        # a truncated ckpt_latest that resume would crash-loop on.
        _tmp = latest + ".tmp.epoch"   # distinct tmp from the step-saver (.tmp.step), else
        torch.save(ckpt_cpu, _tmp); os.replace(_tmp, latest)   # both race on one tmp -> os.replace FileNotFoundError -> ckpt_epNNN never saved
        if ep % T.ckpt_every == 0:
            _epf = os.path.join(T.out_dir, f"ckpt_ep{ep:03d}.pt")
            _t2 = _epf + ".tmp"; torch.save(ckpt_cpu, _t2); os.replace(_t2, _epf)
            # retention: with virtualized epochs there can be hundreds of ckpt_ep files;
            # keep only the most recent CKPT_KEEP_RECENT (env, default keep-all) so the
            # per-user disk quota doesn't fill up.
            _keep = int(os.environ.get("CKPT_KEEP_RECENT", "0") or "0")
            if _keep > 0:
                _eps = sorted(glob.glob(os.path.join(T.out_dir, "ckpt_ep*.pt")))
                for _f in _eps[:-_keep]:
                    try: os.remove(_f)
                    except OSError: pass
        print(f"  [val ep{ep}] saved ckpt", flush=True)

    val_thread = None
    step_save_thread = None

    # -- training loop ---------------------------------------------------------
    # grad_accum: micro-batch of T.batch_size, optimizer steps every `accum`
    # micro-batches -> effective batch = batch_size * world * accum. global_step
    # counts MICRO-batches (so disc_start_step/total_steps/val timing are unchanged
    # and resume stays consistent); scheduler steps per micro-batch so the LR-vs-
    # progress curve is identical regardless of accum (len(loader) scales with bs).
    accum = max(1, T.grad_accum_steps)
    optimizer.zero_grad(set_to_none=True)
    disc_optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, T.epochs):
        train_sampler.set_epoch(epoch)
        if combine_has_params:
            combine_ddp.train()
        decoder_ddp.train()
        t0 = time.time()
        for micro_idx, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            with torch.no_grad():
                layer_tokens = encode_layers(imgs)
            use_gan = (global_step >= disc_start_step) and (L.gan.disc_weight > 0)
            is_accum_step = ((micro_idx + 1) % accum == 0)

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
                with autocast_ctx:
                    fake_aug = disc_aug.aug(x_rec * 2 - 1)   # full-batch GAN (matches official RAEv2)
                    logits_fake, _ = disc_ddp(fake_aug, None)
                loss_gan = vanilla_g_loss(logits_fake)
                last_layer = next(reversed(list(decoder_ddp.module.parameters())))
                adp_w = calculate_adaptive_weight(loss_rec, loss_gan, last_layer).clamp(0, 1e4).detach()
            else:
                adp_w = torch.tensor(0.0, device=device)

            sig_w = L.sigreg.weight if use_sig else 0.0
            loss = loss_rec + sig_w * loss_sig + L.gan.disc_weight * adp_w * loss_gan
            (loss / accum).backward()
            if is_accum_step:
                if T.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, T.clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if use_gan:
                disc_ddp.train()
                with autocast_ctx:
                    real_aug = disc_aug.aug(imgs * 2 - 1)            # full-batch GAN (matches official RAEv2)
                    fake_aug = disc_aug.aug(x_rec.detach() * 2 - 1)
                    logits_real, _ = disc_ddp(real_aug, None)
                    logits_fake, _ = disc_ddp(fake_aug, None)
                    loss_d = hinge_d_loss(logits_real, logits_fake)
                (loss_d / accum).backward()
                if is_accum_step:
                    disc_optimizer.step()
                    disc_optimizer.zero_grad(set_to_none=True)

            if is_accum_step:
                update_ema(ema_combine, combine, T.ema_decay)
                update_ema(ema_dec, decoder, T.ema_decay)
            global_step += 1

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

            # -- step-based ckpt_latest (preemption resilience) ------------------
            # epoch=`epoch` (not +1) so resume redoes the current epoch from its start
            # but with the model/optimizer/scheduler/global_step restored (no weight loss).
            if is_main and ckpt_every_steps > 0 and global_step % ckpt_every_steps == 0:
                if step_save_thread is not None:
                    step_save_thread.join()
                _snap = _to_cpu({
                    "epoch": epoch, "global_step": global_step, "layers": layers,
                    "combine": combine.state_dict(), "decoder": decoder.state_dict(),
                    "ema_combine": None if no_ema else ema_combine.state_dict(),
                    "ema_dec":     None if no_ema else ema_dec.state_dict(),
                    "disc": disc.state_dict(), "optimizer": optimizer.state_dict(),
                    "disc_optimizer": disc_optimizer.state_dict(), "scheduler": scheduler.state_dict()})
                def _save_latest(s):
                    _t = latest + ".tmp.step"; torch.save(s, _t); os.replace(_t, latest)
                step_save_thread = threading.Thread(target=_save_latest, args=(_snap,), daemon=True)
                step_save_thread.start()
                print(f"  [step-ckpt] saved ckpt_latest @ step {global_step}", flush=True)

        if is_main:
            print(f"Epoch {epoch+1}/{T.epochs}  time={time.time()-t0:.0f}s", flush=True)

        # -- async val + ckpt ---------------------------------------------------
        # rank 0 takes a frozen snapshot (EMA deepcopy + CPU-cloned ckpt), hands
        # it to a daemon thread, then hits the barrier immediately. val + save run
        # concurrently with the next epoch's training -> no epoch-boundary stall.
        if is_main:
            if val_thread is not None:
                val_thread.join()                 # serialize: prev epoch's val/save done
            sn_combine = deepcopy(ema_combine).eval()
            sn_dec = deepcopy(ema_dec).eval()
            ckpt_cpu = _to_cpu({
                "epoch": epoch + 1, "global_step": global_step, "layers": layers,
                "combine": combine.state_dict(), "decoder": decoder.state_dict(),
                "ema_combine": ema_combine.state_dict(), "ema_dec": ema_dec.state_dict(),
                "disc": disc.state_dict(), "optimizer": optimizer.state_dict(),
                "disc_optimizer": disc_optimizer.state_dict(), "scheduler": scheduler.state_dict()})
            do_probe = (P.loo_solo == "every") or (P.loo_solo == "final" and epoch + 1 == T.epochs)
            val_thread = threading.Thread(
                target=_async_val_save,
                args=(sn_combine, sn_dec, ckpt_cpu, epoch + 1, global_step, do_probe),
                daemon=True)
            val_thread.start()

        # cheap sync point (rank 0 only snapshotted, no slow work) so no rank drifts
        dist.barrier()

    if is_main and val_thread is not None:
        val_thread.join()                         # finish the last epoch's val + ckpt
    if cfg.wandb.enabled and is_main:
        import wandb; wandb.finish()
    dist.destroy_process_group()
    if is_main:
        print("Done.")


if __name__ == "__main__":
    main()
