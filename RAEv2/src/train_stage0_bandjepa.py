#!/usr/bin/env python3
"""Stage-0: DECODER-LESS 3-band depth-JEPA pretraining of the DepthAttnCombine fusion.

Promotes the semantic-rent tax (src/stage1/semantic_rent.py) to the ENTIRE
objective. No decoder, no L1/LPIPS/GAN: the fusion latent z_b is trained ONLY to
linearly predict the standardized, stop-grad mean of strided frozen-encoder LAYER
bands (shallow/mid/deep). Random per-sample layer drop (stratified mask) turns a
band whose layers were dropped into an imputation-from-survivors task -> masked
modeling along DEPTH, in latent space (JEPA), not pixel-MAE.

vs src/train_fusion_jepa.py: that Stage-0 used a subset->full JEPA predictor +
frozen-L23 grounding + SIGReg. This one replaces the predictor/grounding scaffold
with N explicit band heads (BandPredictHeads); SIGReg is kept (optional) purely to
keep the latent isotropic-Gaussian = diffusable for the downstream DiT.

The checkpoint's `combine` (and `ema_combine`, if ema_decay>0) is the ONLY thing
Stage-1/DiT consumes -- band heads are throwaway scaffolds. Key names match
train_decoder.py's `init_from` warm-start.

Usage:
    torchrun --nproc_per_node=8 src/train_stage0_bandjepa.py \
        --config configs/stage0/bandjepa-depthattn-k23.yaml
"""

import sys, os, math, argparse, time, copy
from dataclasses import dataclass, field
from typing import Any, Dict, List
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from torchvision import transforms

from encoders.vision_encoder import create_encoder
from stage1.mask_cond import sample_stratified_masks
from stage1.band_jepa import BandPredictHeads
from overfit_sigreg import sigreg_loss, gaussian_diag
from utils.model_utils import get_obj_from_str


# --------------------------------------------------------------------------- config
@dataclass
class CombineCfg:
    target: str = "stage1.combine.DepthAttnCombine"
    params: Dict[str, Any] = field(default_factory=dict)     # must include layers, dim


@dataclass
class DataCfg:
    data_dir: str = ""
    image_size: int = 256
    num_workers: int = 8


@dataclass
class BandsCfg:
    # positions into combine.layers; negatives count from the end. Defaults stride
    # the 23 layers into 3 even bands of 4: shallow L7/5/3/1, mid L15/13/11/9,
    # deep L23/21/19/17 (matches the rent deep/mid sets, plus a shallow band).
    bands: Dict[str, Any] = field(default_factory=lambda: {
        "shallow": [-17, -19, -21, -23], "mid": [-9, -11, -13, -15], "deep": [-1, -3, -5, -7]})
    weights: Dict[str, Any] = field(default_factory=lambda: {
        "shallow": 1.0, "mid": 1.0, "deep": 2.0})
    head_hidden: int = 0             # 0 = LINEAR (weak by design; do NOT widen)
    momentum: float = 0.01


@dataclass
class SigCfg:
    w: float = 0.02                  # 0 disables SIGReg (latent then only band-shaped)
    distributed: bool = True
    scale_by_n: bool = True


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
    ema_decay: float = 0.0           # 0 = no EMA (JEPA lineage); >0 also saves ema_combine
    # stratified layer-drop mask (same sampler as the joint trainer's DepthAttnCombine):
    # full rows == the DiT latent (R_*_full), subset rows == depth-imputation (R_*_sub).
    p_drop: float = 0.5
    full_frac: float = 0.334
    uniform_frac: float = 0.334


@dataclass
class WandbCfg:
    enabled: bool = False
    project: str = "rae-stage0-bandjepa"
    entity: str = ""
    name: str = ""


@dataclass
class Stage0BandConfig:
    combine: CombineCfg = field(default_factory=CombineCfg)
    data: DataCfg = field(default_factory=DataCfg)
    bands: BandsCfg = field(default_factory=BandsCfg)
    sigreg: SigCfg = field(default_factory=SigCfg)
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
    ap.add_argument("--config", required=True, help="Stage-0 band-JEPA training YAML")
    args = ap.parse_args()
    cfg: Stage0BandConfig = OmegaConf.to_object(OmegaConf.merge(
        OmegaConf.structured(Stage0BandConfig), OmegaConf.load(args.config)))
    C, D, BND, S, T = cfg.combine, cfg.data, cfg.bands, cfg.sigreg, cfg.training

    # env overrides (mirror train_decoder.py / train_fusion_jepa.py, so launch
    # scripts scale on any node count)
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

    layers = list(C.params["layers"])
    K = len(layers)
    dim = int(C.params.get("dim", 1024))
    assert C.params.get("out_dim", dim) == dim, "Stage-0 uses the dim-wide combine (out_dim==dim)"
    # We pass an EXPLICIT stratified mask to the combine every step, so the module's
    # own internal sampler is bypassed; keep its params harmless.
    C.params["full_frac"] = 0.0
    if "uniform_frac" in C.params:
        C.params["uniform_frac"] = 0.0
    cls_surr = bool(C.params.get("cls_surrogate", False))

    if is_main:
        os.makedirs(T.out_dir, exist_ok=True)
        print(f"World {world}  batch/GPU {T.batch_size} (global {T.batch_size*world})")
        print(f"combine: {C.target}  K={K} dim={dim}  params={C.params}")
        print(f"bands={BND.bands}  weights={BND.weights}  head_hidden={BND.head_hidden} "
              f"(0=LINEAR)  sigreg.w={S.w}")
        print(f"mask: p_drop={T.p_drop} full_frac={T.full_frac} uniform_frac={T.uniform_frac}  "
              f"ema_decay={T.ema_decay}")
        print("*" * 78)
        print(f"* cls_surrogate={cls_surr} -> THIS DEFINES THE LATENT. "
              + ("mask-gated L23-mean added." if cls_surr else "OFF."))
        print("* Stage-1/stats/probe/DiT must use the SAME cls_surrogate. Change deliberately.")
        print("* Heads are LINEAR by design (weak rent). Widening them hides collapse.")
        print("*" * 78)

    if cfg.wandb.enabled and is_main:
        import wandb
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity or None,
                   name=cfg.wandb.name or None,
                   config=OmegaConf.to_container(OmegaConf.structured(cfg)),
                   tags=["stage0", "bandjepa"])

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

    # -- combine (the trained fusion) + throwaway band heads -------------------
    combine = get_obj_from_str(C.target)(**C.params).to(device)
    assert getattr(combine, "has_params", False), "Stage-0 combine must be learnable"
    heads = BandPredictHeads(dim, K, bands=BND.bands, weights=BND.weights,
                             hidden=BND.head_hidden, momentum=BND.momentum).to(device)

    # optional EMA of the combine (JEPA lineage is no-EMA; kept off by default).
    # buffers (band-head running stats live on `heads`, not here) copy too, harmless.
    ema_combine = None
    if T.ema_decay > 0:
        ema_combine = copy.deepcopy(combine).eval()
        for p in ema_combine.parameters():
            p.requires_grad_(False)

    # combine + heads both trained -> DDP-wrap both. Each is forwarded ONCE per
    # step and every param is used, so keep find_unused_parameters=False /
    # static_graph OFF (the DDP defaults).
    combine_ddp = DDP(combine, device_ids=[local_rank])
    heads_ddp = DDP(heads, device_ids=[local_rank])

    trainable = list(combine.parameters()) + list(heads.parameters())
    if is_main:
        print(f"Trainable: combine {sum(p.numel() for p in combine.parameters())/1e6:.2f}M + "
              f"heads {sum(p.numel() for p in heads.parameters())/1e6:.2f}M")

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

    @torch.no_grad()
    def ema_update():
        d = T.ema_decay
        for pe, pm in zip(ema_combine.parameters(), combine.parameters()):
            pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        for be, bm in zip(ema_combine.buffers(), combine.buffers()):
            be.copy_(bm)

    # -- auto-resume -----------------------------------------------------------
    start_epoch = global_step = 0
    latest = os.path.join(T.out_dir, "ckpt_latest.pt")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        combine.load_state_dict(ck["combine"])
        heads.load_state_dict(ck["heads"])
        if ema_combine is not None and "ema_combine" in ck:
            ema_combine.load_state_dict(ck["ema_combine"])
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
        # `combine` (raw) is the Stage-1 warm-start (key name matches train_decoder.py's
        # init_from). `ema_combine` saved only when EMA is enabled. heads kept for resume.
        d = {"epoch": ep, "global_step": gstep, "layers": layers,
             "combine": combine.state_dict(), "heads": heads.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "config": OmegaConf.to_container(OmegaConf.structured(cfg))}
        if ema_combine is not None:
            d["ema_combine"] = ema_combine.state_dict()
        return _to_cpu(d)

    def _save(ckpt_cpu, ep):
        _tmp = latest + ".tmp"; torch.save(ckpt_cpu, _tmp); os.replace(_tmp, latest)
        if ep % T.ckpt_every == 0:
            _epf = os.path.join(T.out_dir, f"ckpt_ep{ep:03d}.pt")
            _t2 = _epf + ".tmp"; torch.save(ckpt_cpu, _t2); os.replace(_t2, _epf)

    # -- clean per-depth signal: LOO probe (drop EXACTLY one layer) ------------
    # For each layer j, drop only j and read every band's loss -> which layers each
    # band leans on, and whether the deep band survives losing a deep layer. Rank-0
    # local (no collective), on a fixed batch cached from the first step.
    fixed_lt = None

    @torch.no_grad()
    def loo_probe(lt):
        Bp = lt[0].shape[0]
        stkp = torch.stack(lt, 0)
        full = torch.ones(Bp, K, dtype=torch.bool, device=device)
        out = {n: [] for n in heads.names}
        for j in range(K):
            m = full.clone(); m[:, j] = False
            z = combine(lt, mask=m)
            _, parts, _ = heads(z, stkp, mask=m)
            for n in heads.names:
                out[n].append(parts[n].item())
        return out

    # -- training loop ---------------------------------------------------------
    accum = max(1, T.grad_accum_steps)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, T.epochs):
        train_sampler.set_epoch(epoch)
        combine_ddp.train(); heads_ddp.train()
        t0 = time.time()
        for micro_idx, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            B = imgs.shape[0]
            with torch.no_grad():
                layer_tokens = encode_layers(imgs)
                stk = torch.stack(layer_tokens, 0)           # [K,B,N,d] frozen targets
            if is_main and fixed_lt is None:                 # cache once for the LOO probe
                fixed_lt = [t.detach().clone() for t in layer_tokens]
            is_accum_step = ((micro_idx + 1) % accum == 0)

            # stratified mask: full rows (full_frac) = the DiT latent (R_*_full);
            # subset rows = depth-imputation pressure (R_*_sub).
            mask = sample_stratified_masks(B, K, T.p_drop, full_frac=T.full_frac,
                                           uniform_frac=T.uniform_frac, device=device)
            with autocast_ctx:
                z = combine_ddp(layer_tokens, mask=mask)     # [B,N,dim]
                loss_band, parts, metrics = heads_ddp(z, stk, mask=mask)
                loss_sig = z.new_zeros(())
                if S.w > 0:
                    loss_sig = sigreg_loss(z.float().reshape(-1, dim),
                                           distributed=S.distributed, scale_by_n=S.scale_by_n)
                loss = loss_band + S.w * loss_sig
            (loss / accum).backward()
            if is_accum_step:
                if T.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, T.clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema_combine is not None:
                    ema_update()
            scheduler.step()
            global_step += 1

            if global_step % T.log_every == 0 and is_main:
                lr = optimizer.param_groups[0]["lr"]
                zd = gaussian_diag(z)
                mstr = "  ".join(f"{k}={v.item():.3f}" for k, v in sorted(metrics.items()))
                pstr = " ".join(f"{n}={parts[n].item():.3e}" for n in heads.names)
                print(f"  ep{epoch+1} s{global_step}  loss={loss.item():.4e}  "
                      f"band[{pstr}]  sig={loss_sig.item():.3e}  "
                      f"z(kurt={zd['kurt']:.3f} iso_disp={zd['iso_disp']:.3f} dead={zd['dead']})  "
                      f"lr={lr:.2e}", flush=True)
                print(f"    {mstr}", flush=True)
                if cfg.wandb.enabled:
                    import wandb
                    log = {"train/loss": loss.item(), "train/band_total": loss_band.item(),
                           "train/sig": loss_sig.item(), "train/lr": lr,
                           "train/z_kurt": zd["kurt"], "train/z_iso_disp": zd["iso_disp"],
                           "train/z_dead": zd["dead"]}
                    log.update({f"train/band_{n}": parts[n].item() for n in heads.names})
                    log.update({f"train/{k}": v.item() for k, v in metrics.items()})
                    wandb.log(log, step=global_step)

            if is_main and ckpt_every_steps > 0 and global_step % ckpt_every_steps == 0:
                _save(_ckpt(epoch, global_step), epoch)      # epoch (not +1): resume redoes it
                print(f"  [step-ckpt] saved @ step {global_step}", flush=True)

        if is_main:
            print(f"Epoch {epoch+1}/{T.epochs}  time={time.time()-t0:.0f}s", flush=True)
            if fixed_lt is not None:
                loo = loo_probe(fixed_lt)
                for n in heads.names:
                    print(f"  [clean] LOO band-loss[{n}] (drop 1 layer) = ["
                          + " ".join(f"{v:.3e}" for v in loo[n]) + f"]  layers={layers}", flush=True)
                if cfg.wandb.enabled:
                    import wandb
                    wandb.log({f"val/loo_{n}_L{l}": v for n in heads.names
                               for l, v in zip(layers, loo[n])}, step=global_step)
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
