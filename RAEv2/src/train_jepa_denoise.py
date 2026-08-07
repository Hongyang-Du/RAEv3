#!/usr/bin/env python3
"""Stage-0 JEPA denoising-AE feature-fusion pretraining (DECOUPLE fusion from decoder).

Fork of train_fusion_jepa.py that replaces the DepthAttnCombine shared head with a
plain-ViT bottleneck encoder E (stage1/denoise_ae.py). The K frozen DINOv3 layers are
param-free pooled FIRST (masked-mean + gated cls-surrogate), so E never sees the drop
mask and the JEPA latent is a genuinely COMPRESSED code z [B, N, d] (d << dim), not the
dim-wide fusion output.

Recipe (view axis = DEPTH):
  frozen DINOv3 -> K layer tokens (one forward, both views free)
  ctx  view: z_ctx  = E(pool(tokens, mask=random layer subset))   [full_frac forced 0]
  full view: z_full = E(pool(tokens, mask=all-ones))              (live target, shared E)
  L_pred = smooth_l1(predictor(z_ctx, mask), z_full)   (target NOT detached by default)
  L_sig  = per-view SIGReg on z_full, z_ctx in the d-dim bottleneck (isotropic-Gaussian)
  L_gnd  = mse(grounding(z_ctx), frozen L_last pooled) (prices deep-layer / cls info)
  loss   = w_pred*L_pred + w_sig*L_sig + w_gnd*L_gnd

This is the mutual-information flavor: the full latent is ALSO pushed through E and
compressed, and the predictor bridges the two compressed views. predictor + grounding
are throwaway scaffolds (discarded after Stage-0); only E is checkpointed for
Stage-1/DiT (no EMA -- LeJEPA). See stage1/denoise_ae.py.

Usage:
    torchrun --nproc_per_node=8 src/train_jepa_denoise.py \
        --config configs/stage0/jepa-denoise-k23-d256.yaml
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
from stage1.denoise_ae import DenoiseEncoder, DenoisePredictor, GroundingHead, pool_layers
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
    proj: str = "mlp"                # 'mlp' nonlinear projector (JEPA default) | 'linear'
    num_tokens: int = 0              # 0 -> (image_size//16)**2


@dataclass
class PredictorCfg:
    depth: int = 2                   # throwaway; operates in the d-dim bottleneck
    n_heads: int = 8
    mlp_mult: int = 2


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
    uniform_frac: float = 0.0        # ctx sampler stratum; full_frac is FORCED to 0
    cls_surrogate: bool = False
    detach_target: bool = False      # OFF = LeJEPA live target; ON = I-JEPA fallback
    w_pred: float = 1.0
    w_sig: float = 0.02
    w_gnd: float = 1.0


@dataclass
class WandbCfg:
    enabled: bool = False
    project: str = "rae-stage0-jepa"
    entity: str = ""
    name: str = ""


@dataclass
class JepaDenoiseConfig:
    encoder: EncoderCfg = field(default_factory=EncoderCfg)
    predictor: PredictorCfg = field(default_factory=PredictorCfg)
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
    ap.add_argument("--config", required=True, help="Stage-0 JEPA-denoise training YAML")
    args = ap.parse_args()
    cfg: JepaDenoiseConfig = OmegaConf.to_object(OmegaConf.merge(
        OmegaConf.structured(JepaDenoiseConfig), OmegaConf.load(args.config)))
    E, PR, S, D, T = cfg.encoder, cfg.predictor, cfg.sigreg, cfg.data, cfg.training

    # env overrides (mirror train_decoder.py, so launch scripts scale on any node count)
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
        print(f"JEPA-denoise  K={K} dim={dim} d={d}  E(depth={E.depth},proj={E.proj}) "
              f"P(depth={PR.depth})  num_tokens={num_tokens}")
        print(f"loss weights: pred={T.w_pred} sig={T.w_sig} gnd={T.w_gnd}  "
              f"detach_target={T.detach_target}  p_drop={T.p_drop} uniform_frac={T.uniform_frac}")
        print("*" * 78)
        print(f"* cls_surrogate={cls_surr} -> THIS DEFINES THE LATENT. "
              + ("mask-gated L_last mean added (dropped-L_last ctx omits it)."
                 if cls_surr else "OFF: grounding head carries CLS-level info."))
        print(f"* Compression: dim {dim} -> d {d} ({dim/max(d,1):.0f}x). d==dim = no compression.")
        print("* Stage-1/stats/probe/DiT must use the SAME cls_surrogate + d. Change deliberately.")
        print("*" * 78)

    if cfg.wandb.enabled and is_main:
        import wandb
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity or None,
                   name=cfg.wandb.name or None,
                   config=OmegaConf.to_container(OmegaConf.structured(cfg)),
                   tags=["stage0", "jepa-denoise"])

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

    # -- bottleneck encoder E (JEPA head) + throwaway predictor/grounding -------
    enc = DenoiseEncoder(dim=dim, d=d, depth=E.depth, n_heads=E.n_heads,
                         mlp_mult=E.mlp_mult, num_tokens=num_tokens, proj=E.proj).to(device)
    predictor = DenoisePredictor(d, depth=PR.depth, n_heads=PR.n_heads,
                                 mlp_mult=PR.mlp_mult, num_tokens=num_tokens).to(device)
    ground = GroundingHead(in_dim=d, out_dim=dim).to(device)

    # DDP INVARIANT (inherited from train_fusion_jepa.py): keep find_unused_parameters=False
    # and static_graph OFF. enc/ground are each forwarded TWICE per step (ctx + full) before
    # one backward; turning either on would raise "marked ready twice". No EMA anywhere
    # (LeJEPA no-EMA/no-stop-grad): the live z_full target can't collapse because per-view
    # SIGReg pins each latent's marginal to N(0,I).
    enc_ddp = DDP(enc, device_ids=[local_rank])
    predictor_ddp = DDP(predictor, device_ids=[local_rank])
    ground_ddp = DDP(ground, device_ids=[local_rank])

    trainable = list(enc.parameters()) + list(predictor.parameters()) + list(ground.parameters())
    if is_main:
        print(f"Trainable: E {sum(p.numel() for p in enc.parameters())/1e6:.2f}M + "
              f"predictor {sum(p.numel() for p in predictor.parameters())/1e6:.2f}M + "
              f"ground {sum(p.numel() for p in ground.parameters())/1e6:.2f}M")

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
        enc.load_state_dict(ck["encoder"])
        predictor.load_state_dict(ck["predictor"]); ground.load_state_dict(ck["ground"])
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
        # no-EMA: the raw `encoder` weights ARE the Stage-1 warm-start. predictor/ground
        # kept only for resume.
        return _to_cpu({
            "epoch": ep, "global_step": gstep, "layers": layers,
            "cls_surrogate": cls_surr, "dim": dim, "d": d,
            "encoder": enc.state_dict(),
            "predictor": predictor.state_dict(), "ground": ground.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "config": OmegaConf.to_container(OmegaConf.structured(cfg))})

    def _save(ckpt_cpu, ep):
        _tmp = latest + ".tmp"; torch.save(ckpt_cpu, _tmp); os.replace(_tmp, latest)
        if ep % T.ckpt_every == 0:
            _epf = os.path.join(T.out_dir, f"ckpt_ep{ep:03d}.pt")
            _t2 = _epf + ".tmp"; torch.save(ckpt_cpu, _t2); os.replace(_t2, _epf)

    # -- helper: pool + encode (E never sees the mask; pooling erases it) -------
    def embed(lt, mask):
        return enc_ddp(pool_layers(lt, mask, cls_surr))

    # -- diagnostics: pred-loss bucketed by DEEPEST-DROPPED layer (CONFOUNDED) --
    bucket_sum = torch.zeros(K, device=device)     # index = deepest dropped layer position
    bucket_cnt = torch.zeros(K, device=device)
    bucket_ndrop = torch.zeros(K, device=device)   # sum of |dropped| per bucket

    @torch.no_grad()
    def bucket_pred_loss(z_hat, z_tg, mask_ctx):
        err = F.smooth_l1_loss(z_hat.float(), z_tg.float(), reduction="none").mean(dim=(1, 2))  # [B]
        dropped = ~mask_ctx.bool()                                     # [B,K]
        idxK = torch.arange(K, device=device).view(1, K)
        deepest = torch.where(dropped, idxK, torch.full_like(idxK, -1)).max(dim=1).values  # [B], -1 = none
        valid = deepest >= 0
        if valid.any():
            bucket_sum.index_add_(0, deepest[valid], err[valid])
            bucket_cnt.index_add_(0, deepest[valid], torch.ones(int(valid.sum()), device=device))
            bucket_ndrop.index_add_(0, deepest[valid], dropped.sum(1).float()[valid])

    def flush_buckets():
        dist.all_reduce(bucket_sum); dist.all_reduce(bucket_cnt); dist.all_reduce(bucket_ndrop)
        cnt = bucket_cnt.clamp_min(1)
        means, ndrop_mean = (bucket_sum / cnt).tolist(), (bucket_ndrop / cnt).tolist()
        bucket_sum.zero_(); bucket_cnt.zero_(); bucket_ndrop.zero_()
        return means, ndrop_mean

    # -- clean per-depth signal: LOO probe (drop EXACTLY one layer) -------------
    fixed_lt = None

    @torch.no_grad()
    def loo_probe(lt):
        Bp = lt[0].shape[0]
        full = torch.ones(Bp, K, dtype=torch.bool, device=device)
        zf = embed(lt, full)
        errs = []
        for j in range(K):
            m = full.clone(); m[:, j] = False
            errs.append(F.smooth_l1_loss(predictor(embed(lt, m)).float(), zf.float()).item())
        return errs

    # -- training loop ---------------------------------------------------------
    accum = max(1, T.grad_accum_steps)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, T.epochs):
        train_sampler.set_epoch(epoch)
        enc_ddp.train(); predictor_ddp.train(); ground_ddp.train()
        t0 = time.time()
        for micro_idx, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            B = imgs.shape[0]
            with torch.no_grad():
                layer_tokens = encode_layers(imgs)
                gnd_tgt = layer_tokens[-1].mean(1)               # frozen L_last pooled [B,dim]
            if is_main and fixed_lt is None:                     # cache once for the LOO probe
                fixed_lt = [t.detach().clone() for t in layer_tokens]
            is_accum_step = ((micro_idx + 1) % accum == 0)

            # ctx = pure subset stream (full_frac=0); full = explicit all-ones mask so
            # it's the true target regardless of any internal sampling.
            mask_ctx = sample_stratified_masks(B, K, T.p_drop, full_frac=0.0,
                                               uniform_frac=T.uniform_frac, device=device)
            mask_full = torch.ones(B, K, dtype=torch.bool, device=device)

            with autocast_ctx:
                z_ctx = embed(layer_tokens, mask_ctx)            # E(pool(dropped)) [B,N,d]
                z_full = embed(layer_tokens, mask_full)          # E(pool(full))    [B,N,d]
                z_hat = predictor_ddp(z_ctx)
                z_tg = z_full.detach() if T.detach_target else z_full
                loss_pred = F.smooth_l1_loss(z_hat.float(), z_tg.float())
                # grounding prices deep/cls info. For z_ctx the L_last-KEPT samples make the
                # task trivial; stratify by L_last kept/dropped and weight the two strata
                # EQUALLY so the dropped signal isn't diluted. Also anchor z_full directly.
                err_ctx = ((ground_ddp(z_ctx).float() - gnd_tgt.float()) ** 2).mean(-1)   # [B]
                llast_kept = mask_ctx[:, -1].bool()
                gnd_parts = [err_ctx[m].mean() for m in (llast_kept, ~llast_kept) if m.any()]
                loss_gnd_ctx = torch.stack(gnd_parts).mean()
                loss_gnd_full = ((ground_ddp(z_full).float() - gnd_tgt.float()) ** 2).mean()
                loss_gnd = loss_gnd_ctx + loss_gnd_full
                # per-view SIGReg (LeJEPA no-EMA anti-collapse) in the d-dim bottleneck: pin
                # EACH of z_full and z_ctx to N(0,I) over its own B*N samples, pooled cross-rank.
                loss_sig = 0.5 * (sigreg_loss(z_full.float().reshape(-1, d),
                                              distributed=S.distributed, scale_by_n=S.scale_by_n)
                                  + sigreg_loss(z_ctx.float().reshape(-1, d),
                                                distributed=S.distributed, scale_by_n=S.scale_by_n))
            loss = T.w_pred * loss_pred + T.w_sig * loss_sig + T.w_gnd * loss_gnd
            (loss / accum).backward()
            if is_accum_step:
                if T.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, T.clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            bucket_pred_loss(z_hat, z_tg, mask_ctx)
            global_step += 1

            if global_step % T.log_every == 0:
                means, ndrop_mean = flush_buckets()              # collective: ALL ranks call
                if is_main:
                    lr = optimizer.param_groups[0]["lr"]
                    zd = gaussian_diag(z_full)
                    gk = err_ctx[llast_kept].mean().item() if llast_kept.any() else float("nan")
                    gd = err_ctx[~llast_kept].mean().item() if (~llast_kept).any() else float("nan")
                    print(f"  ep{epoch+1} s{global_step}  loss={loss.item():.4e}  "
                          f"pred={loss_pred.item():.4e}  gnd={loss_gnd.item():.4e}  "
                          f"sig={loss_sig.item():.4e}  z_full(kurt={zd['kurt']:.3f} "
                          f"iso_disp={zd['iso_disp']:.3f} dead={zd['dead']})  lr={lr:.2e}", flush=True)
                    print(f"    gnd Llast[kept={gk:.3e} dropped={gd:.3e}]  full={loss_gnd_full.item():.3e}", flush=True)
                    print("    [confounded] pred-loss by deepest-dropped Lpos = ["
                          + " ".join(f"{m:.3e}" for m in means) + f"]  <|dropped|>=["
                          + " ".join(f"{n:.1f}" for n in ndrop_mean) + "]", flush=True)
                    if cfg.wandb.enabled:
                        import wandb
                        wandb.log({"train/loss": loss.item(), "train/pred": loss_pred.item(),
                                   "train/gnd": loss_gnd.item(), "train/gnd_full": loss_gnd_full.item(),
                                   "train/gnd_Llastkept": gk, "train/gnd_Llastdropped": gd,
                                   "train/sig": loss_sig.item(),
                                   "train/z_kurt": zd["kurt"], "train/z_iso_disp": zd["iso_disp"],
                                   "train/z_dead": zd["dead"], "train/lr": lr,
                                   **{f"train/predloss_droppos_{i}": means[i] for i in range(K)}},
                                  step=global_step)

            if is_main and ckpt_every_steps > 0 and global_step % ckpt_every_steps == 0:
                _save(_ckpt(epoch, global_step), epoch)          # epoch (not +1): resume redoes it
                print(f"  [step-ckpt] saved @ step {global_step}", flush=True)

        if is_main:
            print(f"Epoch {epoch+1}/{T.epochs}  time={time.time()-t0:.0f}s", flush=True)
            if fixed_lt is not None:
                loo = loo_probe(fixed_lt)
                print("  [clean] LOO pred-loss (drop exactly 1 layer) = ["
                      + " ".join(f"{v:.3e}" for v in loo) + f"]  layers={layers}", flush=True)
                if cfg.wandb.enabled:
                    import wandb
                    wandb.log({f"val/loo_predloss_L{l}": v for l, v in zip(layers, loo)},
                              step=global_step)
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
