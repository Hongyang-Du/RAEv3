#!/usr/bin/env python3
"""Stage-0 MAE denoising-AE feature-fusion pretraining (DECOUPLE fusion from decoder).

Trains a bottleneck encoder E by RECONSTRUCTING the full-layer pooled latent from a
layer-dropout-corrupted one -- no decoder/L1/LPIPS/GAN, no predictor, no live target.

Recipe (view axis = DEPTH):
  frozen DINOv3 -> K layer tokens (one forward)
  input  : z0_in  = pool(tokens, mask=random layer subset)   [B,N,dim]  (corrupted)
  target : z0_tgt = pool(tokens, mask=all-ones)              [B,N,dim]  (FIXED, frozen)
  z      = E(z0_in)                                          [B,N,d]    (bottleneck)
  recon  = D_mae(z)                                          [B,N,dim]
  L_rec  = smooth_l1(recon, z0_tgt)
  L_sig  = SIGReg(z)     (optional: shape the bottleneck toward N(0,I) for DiT; w_sig=0
                          -> pure MAE)
  loss   = w_recon*L_rec + w_sig*L_sig

D_mae is a throwaway scaffold (discarded after Stage-0); only E is checkpointed for
Stage-1/DiT (no EMA). full_frac>0 keeps the clean full-input path in-distribution
(Stage-1/eval feed pool(full), no drop). See stage1/denoise_ae.py.

Usage:
    torchrun --nproc_per_node=8 src/train_mae_denoise.py \
        --config configs/stage0/mae-denoise-k23-d256.yaml
"""

import sys, os, math, argparse, time
from dataclasses import dataclass, field
from typing import Any, Dict, List
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from torchvision import transforms

from encoders.vision_encoder import create_encoder
from stage1.mask_cond import sample_stratified_masks
from stage1.denoise_ae import DenoiseEncoder, MAEDecoder, pool_layers
from overfit_sigreg import sigreg_loss, gaussian_diag


# --------------------------------------------------------------------------- config
@dataclass
class EncoderCfg:
    layers: List[int] = field(default_factory=list)
    dim: int = 1024
    d: int = 256                     # bottleneck width (the experiment's variable)
    depth: int = 6
    n_heads: int = 8
    mlp_mult: int = 4
    proj: str = "linear"             # 'linear' clean latent (MAE default) | 'mlp'
    num_tokens: int = 0              # 0 -> (image_size//16)**2


@dataclass
class DecoderCfg:
    depth: int = 6
    n_heads: int = 8
    mlp_mult: int = 4


@dataclass
class SigCfg:
    distributed: bool = True
    scale_by_n: bool = True


@dataclass
class DataCfg:
    data_dir: str = ""
    image_size: int = 256
    num_workers: int = 8


@dataclass
class TrainCfg:
    out_dir: str = ""
    epochs: int = 10
    warmup_epochs: int = 1
    batch_size: int = 64
    lr: float = 2e-4
    seed: int = 0
    grad_accum_steps: int = 1
    clip_grad: float = 1.0
    precision: str = "bf16"
    log_every: int = 50
    ckpt_every: int = 1
    p_drop: float = 0.75
    full_frac: float = 0.15          # keep the clean full-input path in-distribution
    uniform_frac: float = 0.0
    cls_surrogate: bool = False
    w_recon: float = 1.0
    w_sig: float = 0.02              # 0 -> pure MAE; >0 shapes the bottleneck for DiT


@dataclass
class WandbCfg:
    enabled: bool = False
    project: str = "rae-stage0-mae"
    entity: str = ""
    name: str = ""


@dataclass
class MAEDenoiseConfig:
    encoder: EncoderCfg = field(default_factory=EncoderCfg)
    decoder: DecoderCfg = field(default_factory=DecoderCfg)
    sigreg: SigCfg = field(default_factory=SigCfg)
    data: DataCfg = field(default_factory=DataCfg)
    training: TrainCfg = field(default_factory=TrainCfg)
    wandb: WandbCfg = field(default_factory=WandbCfg)


# --------------------------------------------------------------------------- helpers
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
    ap.add_argument("--config", required=True, help="Stage-0 MAE-denoise training YAML")
    args = ap.parse_args()
    cfg: MAEDenoiseConfig = OmegaConf.to_object(OmegaConf.merge(
        OmegaConf.structured(MAEDenoiseConfig), OmegaConf.load(args.config)))
    E, DEC, S, D, T = cfg.encoder, cfg.decoder, cfg.sigreg, cfg.data, cfg.training

    # env overrides (mirror train_fusion_jepa.py so launch scripts scale on any node count)
    if os.environ.get("BATCH_SIZE_OVERRIDE"): T.batch_size = int(os.environ["BATCH_SIZE_OVERRIDE"])
    if os.environ.get("LR_OVERRIDE"): T.lr = float(os.environ["LR_OVERRIDE"])
    if os.environ.get("EPOCHS_OVERRIDE"): T.epochs = int(os.environ["EPOCHS_OVERRIDE"])
    if os.environ.get("WARMUP_OVERRIDE"): T.warmup_epochs = int(os.environ["WARMUP_OVERRIDE"])
    if os.environ.get("OUT_DIR_OVERRIDE"): T.out_dir = os.environ["OUT_DIR_OVERRIDE"]
    if os.environ.get("NUM_WORKERS_OVERRIDE"): D.num_workers = int(os.environ["NUM_WORKERS_OVERRIDE"])
    ckpt_every_steps = int(os.environ.get("CKPT_EVERY_STEPS", "0") or "0")

    from datetime import timedelta
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(T.seed + rank)
    torch.backends.cudnn.benchmark = True
    is_main = rank == 0

    layers = list(E.layers)
    K = len(layers)
    dim, d = E.dim, E.d
    num_tokens = E.num_tokens or (D.image_size // 16) ** 2
    cls_surr = bool(T.cls_surrogate)
    if is_main:
        os.makedirs(T.out_dir, exist_ok=True)
        print(f"World {world}  batch/GPU {T.batch_size} (global {T.batch_size*world})")
        print(f"MAE-denoise  K={K} dim={dim} d={d}  E(depth={E.depth},proj={E.proj}) "
              f"D_mae(depth={DEC.depth})  num_tokens={num_tokens}")
        print(f"p_drop={T.p_drop} full_frac={T.full_frac} uniform_frac={T.uniform_frac}  "
              f"w_recon={T.w_recon} w_sig={T.w_sig}")
        print("*" * 78)
        print(f"* cls_surrogate={cls_surr} -> DEFINES THE LATENT (target = pool(full)). "
              + ("mask-gated L_last mean." if cls_surr else "OFF."))
        print(f"* Compression: dim {dim} -> d {d} ({dim/max(d,1):.0f}x). d==dim = no compression.")
        print("* Stage-1/stats/probe/DiT must use the SAME cls_surrogate + d. Change deliberately.")
        print("*" * 78)

    if cfg.wandb.enabled and is_main:
        import wandb
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity or None,
                   name=cfg.wandb.name or None,
                   config=OmegaConf.to_container(OmegaConf.structured(cfg)),
                   tags=["stage0", "mae-denoise"])

    # -- data ------------------------------------------------------------------
    train_loader, train_sampler = make_loader(
        D.data_dir, D.image_size, T.batch_size, D.num_workers, world, rank)
    if is_main:
        print(f"Train: {len(train_loader.dataset)} images  ({len(train_loader)} steps/epoch)")

    # -- frozen DINOv3 encoder -------------------------------------------------
    layers_str = ".".join(str(x) for x in layers)
    encoder = create_encoder(f"dinov3mls-vit-l16[layers={layers_str}]",
                             device=device, resolution=D.image_size)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    enc_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    enc_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def encode_layers(imgs01):
        imgs_norm = (imgs01 - enc_mean) / enc_std
        return list(encoder.model.get_intermediate_layers(
            imgs_norm, n=encoder.layer_indices, reshape=False,
            return_class_token=False, norm=True))

    # -- bottleneck encoder E + throwaway MAE decoder --------------------------
    enc = DenoiseEncoder(dim=dim, d=d, depth=E.depth, n_heads=E.n_heads,
                         mlp_mult=E.mlp_mult, num_tokens=num_tokens, proj=E.proj).to(device)
    dec = MAEDecoder(dim=dim, d=d, depth=DEC.depth, n_heads=DEC.n_heads,
                     mlp_mult=DEC.mlp_mult, num_tokens=num_tokens).to(device)
    enc_ddp = DDP(enc, device_ids=[local_rank])
    dec_ddp = DDP(dec, device_ids=[local_rank])
    trainable = list(enc.parameters()) + list(dec.parameters())
    if is_main:
        print(f"Trainable: E {sum(p.numel() for p in enc.parameters())/1e6:.2f}M + "
              f"D_mae {sum(p.numel() for p in dec.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(trainable, lr=T.lr, betas=(0.9, 0.95), weight_decay=0.0)
    total_steps = T.epochs * len(train_loader)
    warmup_steps = T.warmup_epochs * len(train_loader)

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        pr = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * pr))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(T.precision == "bf16"))

    # -- auto-resume -----------------------------------------------------------
    start_epoch = global_step = 0
    latest = os.path.join(T.out_dir, "ckpt_latest.pt")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        enc.load_state_dict(ck["encoder"]); dec.load_state_dict(ck["decoder"])
        optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"])
        start_epoch, global_step = ck["epoch"], ck["global_step"]
        if is_main:
            print(f"Resumed from {latest} (epoch={start_epoch}, step={global_step})")
        del ck

    def _to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.detach().to("cpu", copy=True)
        if isinstance(obj, dict):
            return {k: _to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_cpu(v) for v in obj]
        return obj

    def _ckpt(ep, gstep):
        # no-EMA: the raw `encoder` weights ARE the Stage-1 warm-start. decoder kept for resume.
        return _to_cpu({
            "epoch": ep, "global_step": gstep, "layers": layers,
            "cls_surrogate": cls_surr, "dim": dim, "d": d,
            "encoder": enc.state_dict(), "decoder": dec.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "config": OmegaConf.to_container(OmegaConf.structured(cfg))})

    def _save(ckpt_cpu, ep):
        _tmp = latest + ".tmp"; torch.save(ckpt_cpu, _tmp); os.replace(_tmp, latest)
        if ep % T.ckpt_every == 0:
            _epf = os.path.join(T.out_dir, f"ckpt_ep{ep:03d}.pt")
            _t2 = _epf + ".tmp"; torch.save(ckpt_cpu, _t2); os.replace(_t2, _epf)

    # -- diagnostics: recon loss bucketed by |dropped| -------------------------
    bucket_sum = torch.zeros(K + 1, device=device)   # index = number of dropped layers
    bucket_cnt = torch.zeros(K + 1, device=device)

    @torch.no_grad()
    def bucket_recon(recon, z0_tgt, mask):
        err = F.smooth_l1_loss(recon.float(), z0_tgt.float(), reduction="none").mean(dim=(1, 2))  # [B]
        ndrop = (~mask.bool()).sum(1)                # [B] in 0..K
        bucket_sum.index_add_(0, ndrop, err)
        bucket_cnt.index_add_(0, ndrop, torch.ones_like(err))

    def flush_buckets():
        dist.all_reduce(bucket_sum); dist.all_reduce(bucket_cnt)
        means = (bucket_sum / bucket_cnt.clamp_min(1)).tolist()
        bucket_sum.zero_(); bucket_cnt.zero_()
        return means

    # -- training loop ---------------------------------------------------------
    accum = max(1, T.grad_accum_steps)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, T.epochs):
        train_sampler.set_epoch(epoch)
        enc_ddp.train(); dec_ddp.train()
        t0 = time.time()
        for micro_idx, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            B = imgs.shape[0]
            with torch.no_grad():
                layer_tokens = encode_layers(imgs)
            is_accum_step = ((micro_idx + 1) % accum == 0)

            # corruption mask for the INPUT (target is always the full pool). full_frac>0
            # so the clean full-input path (what Stage-1/eval feed) stays in-distribution.
            mask_in = sample_stratified_masks(B, K, T.p_drop, full_frac=T.full_frac,
                                              uniform_frac=T.uniform_frac, device=device)
            mask_full = torch.ones(B, K, dtype=torch.bool, device=device)

            with autocast_ctx:
                z0_in = pool_layers(layer_tokens, mask_in, cls_surr)     # [B,N,dim] corrupted
                z0_tgt = pool_layers(layer_tokens, mask_full, cls_surr)  # [B,N,dim] FIXED target
                z = enc_ddp(z0_in)                                       # [B,N,d] bottleneck
                recon = dec_ddp(z)                                       # [B,N,dim]
                loss_recon = F.smooth_l1_loss(recon.float(), z0_tgt.float())
                loss_sig = sigreg_loss(z.float().reshape(-1, d),
                                       distributed=S.distributed, scale_by_n=S.scale_by_n) \
                    if T.w_sig > 0 else z.new_zeros(())
            loss = T.w_recon * loss_recon + T.w_sig * loss_sig
            (loss / accum).backward()
            if is_accum_step:
                if T.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, T.clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            bucket_recon(recon, z0_tgt, mask_in)
            global_step += 1

            if global_step % T.log_every == 0:
                means = flush_buckets()                  # collective: ALL ranks call
                if is_main:
                    lr = optimizer.param_groups[0]["lr"]
                    zd = gaussian_diag(z)
                    print(f"  ep{epoch+1} s{global_step}  loss={loss.item():.4e}  "
                          f"recon={loss_recon.item():.4e}  sig={loss_sig.item():.4e}  "
                          f"z(kurt={zd['kurt']:.3f} iso_disp={zd['iso_disp']:.3f} "
                          f"dead={zd['dead']})  lr={lr:.2e}", flush=True)
                    print("    recon by |dropped| = ["
                          + " ".join(f"{m:.3e}" for m in means) + "]", flush=True)
                    if cfg.wandb.enabled:
                        import wandb
                        wandb.log({"train/loss": loss.item(), "train/recon": loss_recon.item(),
                                   "train/sig": loss_sig.item(), "train/z_kurt": zd["kurt"],
                                   "train/z_iso_disp": zd["iso_disp"], "train/z_dead": zd["dead"],
                                   "train/lr": lr,
                                   **{f"train/recon_ndrop_{i}": means[i] for i in range(K + 1)}},
                                  step=global_step)

            if is_main and ckpt_every_steps > 0 and global_step % ckpt_every_steps == 0:
                _save(_ckpt(epoch, global_step), epoch)
                print(f"  [step-ckpt] saved @ step {global_step}", flush=True)

        if is_main:
            print(f"Epoch {epoch+1}/{T.epochs}  time={time.time()-t0:.0f}s", flush=True)
            _save(_ckpt(epoch + 1, global_step), epoch + 1)
            print(f"  saved ckpt ep{epoch+1}", flush=True)
        dist.barrier()

    if cfg.wandb.enabled and is_main:
        import wandb; wandb.finish()
    dist.destroy_process_group()
    if is_main:
        print("Done.")


if __name__ == "__main__":
    main()
