#!/usr/bin/env python3
"""
Multi-GPU DDP training: raev2 + LEARNABLE SOFTGATE — nothing else changed.

raev2 baseline (train_decoder_mls.py recipe: L1+LPIPS+GAN, NO projector MLP,
NO SIGReg constraint) with exactly ONE change: the fixed MLS mean is replaced by
a learnable softmax gate, z = Σ softmax(gate)_i · layer_i (no CLS surrogate —
it belongs to the combine being replaced and would inject fixed L23 content).
SIGReg is computed for LOGGING ONLY (run with --sigreg-w 0).

The purest collapse test: at K7 such gates collapsed to the shallowest layer
(L11, w=0.98 by ep3); does the same happen at K=24 (collapse target = L0)?
Per-epoch LOO/solo probes map per-layer reliance for a direct comparison with
the dropmean_bn_all24 twin. --layer-drop (default 0) can re-add per-sample
layer dropout to reproduce the old softgate mechanics.

Inherited plumbing notes:
  1. Projector MLP norm: pre-LayerNorm residual -> LeWM recipe
     (fc1 -> BatchNorm1d(hidden, over B*N token samples) -> GELU -> fc2, + skip;
     output is a bare Linear so SIGReg sees an unconstrained distribution).
  2. SIGReg computed GLOBALLY across ranks (differentiable all-reduce of the ECF
     means) instead of per-rank.
Intended first use: ALL DINOv3-L layers (--layers 0..23) to map every layer's
contribution via the built-in per-epoch LOO/solo probes.

Drop-mean variant of train_decoder_mls_nogate_sigreg.py: the K MLS layers are combined
by an EQUAL-WEIGHT mean over a RANDOM SUBSET of layers (per-sample layer dropout,
--layer-drop). There is NO learnable gate at all — any learnable per-layer weight
(free sigmoid OR softmax+dropout) collapses to the shallowest layer under the
recon+GAN objective, because with a weight to learn the optimum IS that layer. Here
nothing can be biased: all layers enter with weight 1/|subset|, and the decoder is
forced to reconstruct from every random subset — including subsets WITHOUT the
shallowest layer — so it must learn to use the deep layers. Eval uses the full
equal-weight mean (= RAEv2 / nogate combine); stats match training because the
subset combine is a renormalized mean, not a sum.

Architecture:
  DINOv3-L (frozen, layers [11,13,15,17,19,21,23])
       v
  drop-mean combine (no params):                   z0 [B, 256, 1024]
    train: mean over random per-sample layer subset
    eval:  mean over all K layers
       v
  Projector (learnable, per-token residual MLP)    z  [B, 256, latent]   <- SIGReg here
       v
  ViT Decoder (trained from scratch)               x_rec [B, 3, 256, 256]

Loss: L1 + LPIPS + GAN + sigreg_w * SIGReg
  - recon/LPIPS/GAN train projector + decoder (end-to-end)
  - SIGReg pushes z toward N(0, I); its gradient flows into the projector only
    (encoder + MLS are frozen/param-free).

Usage:
    torchrun --nproc_per_node=8 src/train_decoder_mls_dropmean_sigreg.py \
        --data /datasets/imagenet-256 \
        --epochs 100 --batch-size 64 --sigreg-w 1 --layer-drop 0.3 --wandb
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

from models.spatial_attn_res import RAEV2_LAYERS
from overfit_sigreg import sigreg_loss, gaussian_diag, psnr
from stage1.disc import DinoDiscriminator, hinge_d_loss, vanilla_g_loss, calculate_adaptive_weight
from stage1.disc.diffaug import DiffAug


# --- Projector ----------------------------------------------------------------

class MLSProjector(nn.Module):
    """PLAIN softmax-gated combine: z = Σ softmax(gate)_i · layer_i. Nothing else.

    No projector MLP, no BN, no SIGReg — the purest collapse test: a normal MLS
    latent where the only learnable thing between the frozen encoder and the
    decoder is one weight per layer (+ per-sample layer dropout, as in the K7
    softgate). K7 evidence: such gates collapse to the shallowest layer
    (w=[0.98 0.02 0 ...] by ep3). This run asks whether that repeats at K=24
    (collapse target = L0). No CLS surrogate either — it would inject fixed
    last-layer content and mask what the gate itself chooses.

    forward(layer_tokens, idx=None): idx (positions into the FULL gate) lets the
    eval probes decode arbitrary layer subsets — softmax weights renormalized
    over the subset, mirroring how LOO/solo work for dropmean.
    """
    def __init__(self, n_layers: int, dim: int = 1024, out_dim: int = 1024,
                 mult: int = 4, p_drop: float = 0.0):
        super().__init__()
        self.p_drop = p_drop                                 # 0 = pure gate (raev2 + gate, nothing else)
        self.gate = nn.Parameter(torch.zeros(n_layers))     # softmax logits, init -> uniform 1/K

    def forward(self, layer_tokens, idx=None) -> torch.Tensor:  # K x [B,N,dim] -> [B,N,dim]
        stk = torch.stack(layer_tokens, dim=0)              # [K, B, N, dim]
        K, B = stk.shape[0], stk.shape[1]
        gate_w = torch.softmax(self.gate, dim=0)            # [K_full] convex weights
        if idx is not None:                                 # probe subset: renormalize
            gate_w = gate_w[list(idx)]
            gate_w = gate_w / gate_w.sum().clamp_min(1e-6)
        if self.training and self.p_drop > 0:
            keep = torch.rand(K, B, device=stk.device) > self.p_drop   # per-sample subset
            dead = ~keep.any(0)
            if dead.any():                                  # always keep >= 1 layer
                keep[torch.randint(K, (int(dead.sum()),), device=stk.device), dead] = True
            w = gate_w.view(K, 1) * keep.to(stk.dtype)
            w = w / w.sum(0, keepdim=True).clamp_min(1e-6)  # renormalize over kept subset
            return (w.view(K, B, 1, 1) * stk).sum(0)        # [B, N, dim]
        return (gate_w.view(K, 1, 1, 1) * stk).sum(0)       # full softmax mix (eval)


# --- EMA ----------------------------------------------------------------------

def update_ema(ema, model, decay=0.9995):
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), model.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1 - decay)
        # BN running stats are buffers: copy (they are already running averages);
        # without this the EMA projector keeps init stats and val probes are wrong
        for eb, b in zip(ema.buffers(), model.buffers()):
            eb.copy_(b)


# --- Data ---------------------------------------------------------------------

def make_loader(data_dir, split, image_size, batch_size,
                num_workers=4, world_size=1, rank=0):
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
        from data.partial_imagenet import PartialImageNetDataset
        ds = PartialImageNetDataset(data_dir, split=split, transform=t)
    else:
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


# --- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         required=True)
    parser.add_argument("--image-size",   type=int, default=256)
    parser.add_argument("--epochs",       type=int, default=100)
    parser.add_argument("--batch-size",   type=int, default=64, help="Per-GPU batch size")
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--warmup-epochs",type=int, default=2)
    parser.add_argument("--sigreg-w",     type=float, default=0.1)
    parser.add_argument("--layer-drop",   type=float, default=0.3,
                        help="Per-sample per-layer dropout prob in the drop-mean combine; 0 = plain nogate mean")
    parser.add_argument("--lpips-w",      type=float, default=1.0)
    parser.add_argument("--disc-weight",  type=float, default=0.75)
    parser.add_argument("--disc-start",   type=int,   default=1, help="Epoch to start GAN training")
    parser.add_argument("--disc-ckpt",    type=str,
                        default="pretrained_models/encoders/dino/dino_vit_small_patch8_224.pth")
    parser.add_argument("--ema-decay",    type=float, default=0.9995)
    parser.add_argument("--val-image",    type=str,   default=None,
                        help="Fixed image (its dir is globbed) to reconstruct every epoch")
    parser.add_argument("--clip-grad",    type=float, default=1.0)
    parser.add_argument("--layers",       type=int, nargs="+", default=RAEV2_LAYERS,
                        help="DINOv3 layer indices (default: RAEv2 K7)")
    parser.add_argument("--latent-dim",   type=int, default=1024)
    parser.add_argument("--log-every",    type=int, default=50)
    parser.add_argument("--val-every",    type=int, default=500)
    parser.add_argument("--ckpt-every",   type=int, default=2)
    parser.add_argument("--out-dir",      default="output/train_decoder_mls_softgate_bn_sigreg")
    parser.add_argument("--wandb",        action="store_true")
    parser.add_argument("--wandb-project",default="raev3")
    parser.add_argument("--wandb-entity", default="hongyangd")
    parser.add_argument("--num-workers",  type=int, default=4)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--precision",    type=str, default="fp32", choices=["fp32", "bf16"],
                        help="Training precision. bf16 halves memory usage.")
    args = parser.parse_args()

    # -- DDP init --------------------------------------------------------------
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    is_main = (rank == 0)
    torch.backends.cudnn.benchmark = True

    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"World size: {world_size}  |  batch/GPU: {args.batch_size}"
              f"  |  global batch: {args.batch_size * world_size}")
        print(f"Layers: {args.layers}  |  layer_drop: {args.layer_drop}")

    # -- wandb -----------------------------------------------------------------
    if args.wandb and is_main:
        import wandb
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"decoder-mls-softgate-bn-sigreg-k{len(args.layers)}",
            config={**vars(args), "global_batch": args.batch_size * world_size},
            tags=["decoder", "mls", "sigreg", "softgate", "bn", "stage1", f"k{len(args.layers)}"],
        )

    # -- Data ------------------------------------------------------------------
    train_loader, train_sampler = make_loader(
        args.data, "train", args.image_size, args.batch_size,
        args.num_workers, world_size, rank,
    )
    if is_main:
        print(f"Train: {len(train_loader.dataset)} images")

    # -- DINOv3 encoder (frozen) + raev2 MLS combine ---------------------------
    from encoders.vision_encoder import create_encoder
    layers_str = ".".join(str(l) for l in args.layers)
    encoder = create_encoder(f"dinov3mls-vit-l16[layers={layers_str}]",
                             device=device, resolution=args.image_size)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    enc_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    enc_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    def encode_layers(imgs01: torch.Tensor):
        """imgs01 in [0,1] -> list of K frozen per-layer patch tokens, each [B, 256, 1024].

        These feed MLSProjector, which combines them by a drop-mean (random-subset
        equal-weight mean). Same per-layer extraction the MLS encoder did internally.
        """
        imgs_norm = (imgs01 - enc_mean) / enc_std
        return list(encoder.model.get_intermediate_layers(
            imgs_norm, n=encoder.layer_indices, reshape=False,
            return_class_token=False, norm=True))

    # -- Pre-encode fixed val images (encoder frozen, do once) -----------------
    val_layers_fixed = None
    val_img_orig = None
    if args.val_image and is_main:
        from PIL import Image as PILImage
        _t = transforms.Compose([
            transforms.Resize(args.image_size + 32),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ])
        val_dir   = os.path.dirname(os.path.abspath(args.val_image))
        val_paths = sorted(
            p for p in (glob.glob(os.path.join(val_dir, "*.png")) +
                        glob.glob(os.path.join(val_dir, "*.jpg")))
            if "concat" not in os.path.basename(p).lower()
        )
        if not val_paths:
            val_paths = [args.val_image]
        imgs = [_t(PILImage.open(p).convert("RGB")) for p in val_paths]
        val_img_orig = torch.stack(imgs).to(device)
        with torch.no_grad():
            val_layers_fixed = [t.detach() for t in encode_layers(val_img_orig)]
        print(f"Val images pre-encoded: {len(val_paths)} from {val_dir}", flush=True)

    # -- Projector (learnable) + Decoder (train from scratch) ------------------
    projector = MLSProjector(n_layers=len(args.layers),
                             dim=encoder.hidden_size, out_dim=args.latent_dim,
                             p_drop=args.layer_drop).to(device)

    from omegaconf import OmegaConf
    from stage1.rae import _load_decoder
    s1 = OmegaConf.load("configs/stage2/training/imagenet-dinov3l-k7.yaml").stage_1.params
    decoder = _load_decoder(
        s1.decoder_config_path,
        hidden_size=args.latent_dim,
        patch_size=16,
        num_patches=256,
        pretrained_path=None,
    ).to(device)

    projector_ddp = DDP(projector, device_ids=[local_rank])
    decoder_ddp   = DDP(decoder,   device_ids=[local_rank])

    ema_proj = deepcopy(projector); ema_proj.requires_grad_(False); ema_proj.eval()
    ema_dec  = deepcopy(decoder);   ema_dec.requires_grad_(False);  ema_dec.eval()

    trainable = list(projector.parameters()) + list(decoder.parameters())
    if is_main:
        n = sum(p.numel() for p in trainable) / 1e6
        print(f"Trainable: {n:.1f}M  (projector + decoder)")

    # -- Optimizer + Scheduler -------------------------------------------------
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    # -- LPIPS -----------------------------------------------------------------
    from stage1.disc.lpips import LPIPS as LPIPS_
    lpips_all = LPIPS_().to(device).eval()
    for p in lpips_all.parameters():
        p.requires_grad_(False)

    # -- Discriminator ---------------------------------------------------------
    disc = DinoDiscriminator(
        device=device, dino_ckpt_path=args.disc_ckpt,
        ks=1, recipe="S_8", norm_type="bn", using_spec_norm=True,
    ).to(device)
    from stage1.disc.utils import RandomWindowCrop
    disc.dino_proxy[0].crop = RandomWindowCrop(args.image_size, 224, 9, False)
    disc.dino_proxy[0].original_input_size = args.image_size
    disc_ddp = DDP(disc, device_ids=[local_rank])
    disc_aug = DiffAug(prob=0.5, cutout=True)
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.9))
    disc_start_step = args.disc_start * len(train_loader)
    if is_main:
        n_disc = sum(p.numel() for p in disc.parameters()) / 1e6
        print(f"Discriminator: {n_disc:.1f}M  GAN starts at epoch {args.disc_start}")

    # -- Auto-resume from latest checkpoint ------------------------------------
    start_epoch = 0
    global_step = 0
    latest = os.path.join(args.out_dir, "ckpt_latest.pt")
    if os.path.exists(latest):
        ckpt = torch.load(latest, map_location="cpu")
        projector.load_state_dict(ckpt["projector"])
        decoder.load_state_dict(ckpt["decoder"])
        ema_proj.load_state_dict(ckpt["ema_proj"])
        ema_dec.load_state_dict(ckpt["ema_dec"])
        disc.load_state_dict(ckpt["disc"])
        optimizer.load_state_dict(ckpt["optimizer"])
        disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
        start_epoch  = ckpt["epoch"]
        global_step  = ckpt["global_step"]
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        else:                       # older ckpt without scheduler state: fast-forward
            scheduler.last_epoch = global_step - 1   # so LR doesn't restart its warmup
            scheduler.step()
        if is_main:
            print(f"Resumed from {latest}  (epoch={start_epoch}, step={global_step},"
                  f" lr={optimizer.param_groups[0]['lr']:.2e})")

    # -- Precision -------------------------------------------------------------
    use_bf16 = (args.precision == "bf16")
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)
    if is_main:
        print(f"Precision: {args.precision}")

    # -- Training --------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        projector_ddp.train(); decoder_ddp.train()
        t0 = time.time()

        for step, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)

            # frozen per-layer MLS tokens (no grad / no params)
            with torch.no_grad():
                layer_tokens = encode_layers(imgs)            # K x [B, 256, 1024]

            use_gan = (global_step >= disc_start_step) and (args.disc_weight > 0)

            # -- Step A: Generator (projector + decoder) -----------------------
            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx:
                z = projector_ddp(layer_tokens)               # drop-mean + project (trainable)
                dec_out = decoder_ddp(z, drop_cls_token=False).logits
                x_rec   = (decoder_ddp.module.unpatchify(dec_out)
                           * enc_std + enc_mean).clamp(0, 1)

                loss_l1    = F.l1_loss(x_rec, imgs)
                loss_lpips = lpips_all(x_rec * 2 - 1, imgs * 2 - 1).mean()
            loss_sig   = sigreg_loss(z.float().reshape(-1, args.latent_dim),
                                     distributed=True)      # pooled 8-GPU statistic
            loss_rec   = loss_l1 + args.lpips_w * loss_lpips

            loss_gan = torch.zeros(1, device=device)
            if use_gan:
                disc_ddp.eval()
                half = max(1, x_rec.shape[0] // 2)
                with autocast_ctx:
                    fake_aug = disc_aug.aug(x_rec[:half] * 2 - 1)
                    logits_fake, _ = disc_ddp(fake_aug, None)
                loss_gan = vanilla_g_loss(logits_fake)
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

            # -- Step B: Discriminator -----------------------------------------
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

            update_ema(ema_proj, projector, args.ema_decay)
            update_ema(ema_dec,  decoder,   args.ema_decay)

            global_step += 1

            # -- logging -------------------------------------------------------
            if is_main and global_step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                zd = gaussian_diag(z)
                ps = psnr(x_rec, imgs)
                gate_sig = torch.softmax(projector.gate, dim=0).detach().tolist()
                gate_str = " ".join(f"{v:.2f}" for v in gate_sig)
                print(f"  ep{epoch+1} s{global_step}"
                      f"  loss={loss.item():.4e}"
                      f"  l1={loss_l1.item():.4e}"
                      f"  lpips={loss_lpips.item():.4e}"
                      f"  psnr={ps:.2f}"
                      f"  sig={loss_sig.item():.4e}"
                      f"  gan={loss_gan.item():.4e}"
                      f"  gate=[{gate_str}]"
                      f"  z(mu={zd['mean']:.2f} sd={zd['std']:.2f})"
                      f"  vard(mu={zd['var_mean']:.2f} sd={zd['var_disp']:.3f}) iso={zd['iso_disp']:.3f}"
                      f"  sk={zd['skew']:.3f} ku={zd['kurt']:.2f} d={zd['dead']}"
                      f"  lr={lr:.2e}", flush=True)
                if args.wandb:
                    import wandb
                    wandb.log({
                        "train/loss":   loss.item(),
                        "train/l1":     loss_l1.item(),
                        "train/lpips":  loss_lpips.item(),
                        "train/psnr":   ps,
                        "train/sigreg": loss_sig.item(),
                        "train/gan_g":  loss_gan.item(),
                        "train/adp_w":  adp_w.item(),
                        "train/z_mean":     zd["mean"], "train/z_std": zd["std"],
                        "train/z_var_mean": zd["var_mean"],
                        "train/z_var_disp": zd["var_disp"],
                        "train/z_iso_disp": zd["iso_disp"],
                        "train/z_skew":     zd["skew"], "train/z_kurt": zd["kurt"],
                        "train/z_dead":     zd["dead"],
                        "train/lr":     lr,
                        **{f"train/gate_L{l}": v for l, v in zip(args.layers, gate_sig)},
                    }, step=global_step)

        if is_main:
            print(f"Epoch {epoch+1}/{args.epochs}  time={time.time()-t0:.0f}s", flush=True)

        # -- fixed val images reconstruction -----------------------------------
        if is_main and val_layers_fixed is not None:
            with torch.no_grad():
                val_z   = ema_proj(val_layers_fixed)
                val_dec = ema_dec(val_z, drop_cls_token=False).logits
                val_rec = (ema_dec.unpatchify(val_dec) * enc_std + enc_mean).clamp(0, 1)
            val_ps = psnr(val_rec, val_img_orig)
            print(f"  Val PSNR (EMA): {val_ps:.2f} dB", flush=True)
            if args.wandb:
                import wandb
                wandb.log({"val/psnr": val_ps}, step=global_step)

            # layer-usage probe (LOO/solo, gate-aware): FINAL EPOCH ONLY —
            # per-epoch validation is just the full-mix inference PSNR above.
            # (Gate collapse is tracked continuously via the gate=[...] log line.)
            if (epoch + 1) == args.epochs:
              with torch.no_grad():
                def _subset_psnr(toks, idx):
                    sz  = ema_proj(toks, idx=idx)           # softmax renormalized over subset
                    sd  = ema_dec(sz, drop_cls_token=False).logits
                    sr  = (ema_dec.unpatchify(sd) * enc_std + enc_mean).clamp(0, 1)
                    return psnr(sr, val_img_orig)
                K_val = len(val_layers_fixed)
                loo  = [_subset_psnr([t for j, t in enumerate(val_layers_fixed) if j != i],
                                     [j for j in range(K_val) if j != i])
                        for i in range(K_val)]
                solo = [_subset_psnr([val_layers_fixed[i]], [i]) for i in range(K_val)]
              dloo = [val_ps - v for v in loo]
              print("  Val LOO dPSNR = [" + " ".join(f"{v:+.2f}" for v in dloo)
                    + f"]  layers={args.layers}", flush=True)
              print("  Val solo PSNR = [" + " ".join(f"{v:.2f}" for v in solo) + "]", flush=True)
              if args.wandb:
                  import wandb
                  wandb.log({**{f"val/loo_dpsnr_L{l}": v for l, v in zip(args.layers, dloo)},
                             **{f"val/solo_psnr_L{l}": v for l, v in zip(args.layers, solo)}},
                            step=global_step)

            n = val_rec.shape[0]
            recon_dir = os.path.join(args.out_dir, "val_recon")
            os.makedirs(recon_dir, exist_ok=True)
            save_image(val_rec.cpu(), os.path.join(args.out_dir, "recon_latest.png"), nrow=n)
            if (epoch + 1) % 10 == 0:
                out_path = os.path.join(recon_dir, f"epoch_{epoch+1}.png")
                save_image(val_rec.cpu(), out_path, nrow=n)
                print(f"  Val recon -> {out_path}", flush=True)
                if args.wandb:
                    import wandb
                    wandb.log({"val/recon": wandb.Image(out_path, caption=f"ep={epoch+1}")},
                              step=global_step)

        # -- checkpoint --------------------------------------------------------
        if is_main:
            ckpt = {
                "epoch":          epoch + 1,
                "global_step":    global_step,
                "projector":      projector.state_dict(),
                "decoder":        decoder.state_dict(),
                "ema_proj":       ema_proj.state_dict(),
                "ema_dec":        ema_dec.state_dict(),
                "disc":           disc.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "disc_optimizer": disc_optimizer.state_dict(),
                "scheduler":      scheduler.state_dict(),
                "layers":         args.layers,
            }
            torch.save(ckpt, os.path.join(args.out_dir, "ckpt_latest.pt"))
            if (epoch + 1) % args.ckpt_every == 0:
                ckpt_path = os.path.join(args.out_dir, f"ckpt_ep{epoch+1:03d}.pt")
                torch.save(ckpt, ckpt_path)
                print(f"  Saved {ckpt_path}", flush=True)
            else:
                print(f"  Saved ckpt_latest.pt (ep{epoch+1})", flush=True)

    if args.wandb and is_main:
        import wandb; wandb.finish()
    dist.destroy_process_group()
    if is_main:
        print("Done.")


if __name__ == "__main__":
    main()
