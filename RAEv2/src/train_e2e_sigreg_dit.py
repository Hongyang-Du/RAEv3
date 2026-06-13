#!/usr/bin/env python3
"""End-to-end joint training: representation (projector) + decoder + DiT.

Wiring (user-specified; NO stop-grad, NO EMA target by default — the bet is that
SIGReg pins the latent's distributional geometry and the recon loss pins its
information content, so the generative branch cannot collapse the representation):

  imgs --frozen DINOv3 (K=7 layers)--> projector P --> z (tokens [B,256,1024])
    L_rec = L1(dec(z), GT) + lpips_w*LPIPS(dec(z), GT)        -> P + decoder
    L_sig = SIGReg(z)                                          -> P
  xt = (1-t)*z + t*eps,  t ~ logit-normal + shift 8            (z NOT detached)
  zhat = DiT(xt, t, y)   (x-prediction; IG dual head)
    L_fm  = |zhat - z|^2 / max(t,.05)^2   (full + base head)   -> DiT (+P via target & xt)
    L_pix = L1(dec(zhat), dec(z)) + LPIPS(dec(zhat), dec(z))   -> DiT + decoder + P
            (target is the RECONSTRUCTION, not GT — the deliberate channel through
             which generation gradients shape the projector)

  total = w_rec*L_rec + sigreg_w*L_sig + w_fm*L_fm + w_pix*L_pix

Safety valves (all default OFF = the pure design; flip if the rep collapses —
watch "Val PSNR (EMA)" per epoch: recon degradation is the early collapse alarm):
  --detach-fm-target   stop-grad z in L_fm target (REPA-E style)
  --detach-xt          stop-grad z in the xt interpolation
  --detach-pix-target  stop-grad dec(z) in L_pix
  --pix-t-weight       weight L_pix per-sample by (1-t) (large-t x-preds are
                       conditional means; un-weighted they pull recon toward blur)

Optimizers: projector+decoder AdamW (small lr, expects stage-1 warm-start);
DiT gmuon — identical to the stage-2 baselines for comparability.
Monitoring lines match stage-1/stage-2 formats so plot_val_psnr.py and
plot_dit_progress.py both parse this run's train.log.
No GAN in v1 (stability first). dropmean-style layer dropout is a SWITCH
(--layer-drop, default 0 = plain mean): with a live FM target a random-subset
combine makes the DiT chase per-step target jitter — that cost is part of what
the {nodrop, drop} queue measures. L_pix defaults to 0 (minimal design); EMA is
kept ONLY for the DiT (baseline-FID comparability), projector/decoder eval live.

Usage:
    torchrun --nproc_per_node=8 src/train_e2e_sigreg_dit.py \
        --data /datasets/imagenet-256-full \
        --init-stage1 output_full/train_decoder_mls_nogate_sigreg/ckpt_latest.pt \
        --epochs 10 --batch-size 24 --wandb
"""

import sys, os, math, argparse, time, glob
from copy import deepcopy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torchvision.utils import save_image

from models.spatial_attn_res import RAEV2_LAYERS
from overfit_sigreg import sigreg_loss, gaussian_diag, psnr
from configs.shared import OptimizerConfig
from configs.stage2 import ConditioningArchConfig
from stage2.models.DDT import DiTwDDTHeadIG
from utils.optim_utils import build_optimizer


# --- Projector (nogate: plain mean over K layers + residual MLP) ---------------

class MLSProjector(nn.Module):
    """p_drop > 0 enables dropmean-style per-sample random layer dropout
    (train mode only): Bernoulli keep mask per (layer, sample), >=1 layer kept,
    equal weights renormalized over the kept subset. Eval / EMA always use the
    full mean. NOTE: with dropout on, z is stochastic -> the FM target jitters
    (same image, different z), which raises the irreducible FM loss floor."""

    def __init__(self, dim: int = 1024, out_dim: int = 1024, mult: int = 4,
                 p_drop: float = 0.0):
        super().__init__()
        self.p_drop = p_drop
        self.skip = nn.Linear(dim, out_dim) if dim != out_dim else nn.Identity()
        # LeWM recipe (le-wm config/train/model/lewm.yaml): no pre-norm,
        # BatchNorm on the HIDDEN dim over B*N token samples; output stays
        # a bare Linear so SIGReg sees an unconstrained distribution.
        self.fc1 = nn.Linear(dim, dim * mult)
        self.bn = nn.BatchNorm1d(dim * mult)
        self.fc2 = nn.Linear(dim * mult, out_dim)

    def forward(self, layer_tokens) -> torch.Tensor:        # K x [B, N, dim] -> [B, N, out_dim]
        stk = torch.stack(layer_tokens, dim=0)               # [K, B, N, dim]
        if self.training and self.p_drop > 0:
            K, B = stk.shape[0], stk.shape[1]
            keep = torch.rand(K, B, device=stk.device) > self.p_drop
            dead = ~keep.any(0)
            if dead.any():
                keep[torch.randint(K, (int(dead.sum()),), device=stk.device), dead] = True
            w = keep.float() / keep.float().sum(0, keepdim=True)
            z0 = (w.view(K, B, 1, 1) * stk).sum(0)
        else:
            z0 = stk.mean(0)
        b, n, _ = z0.shape
        h = self.fc1(z0)
        h = self.bn(h.reshape(b * n, -1).float()).reshape(b, n, -1).to(h.dtype)
        return self.skip(z0) + self.fc2(F.gelu(h))

    def load_ln_ckpt(self, sd):
        """Warm-start from an LN-projector state dict (nogate / dropmean
        stage-1): Linear weights carry over, BN stats start fresh."""
        remap = {}
        for k, v in sd.items():
            if k.startswith("ffn.0."):
                remap["fc1." + k.split(".")[-1]] = v
            elif k.startswith("ffn.2."):
                remap["fc2." + k.split(".")[-1]] = v
            elif k.startswith("skip."):
                remap[k] = v
            # norm.{weight,bias} (LayerNorm) intentionally dropped
        missing, unexpected = self.load_state_dict(remap, strict=False)
        print(f"MLSProjector: remapped LN ckpt — fresh: {sorted(missing)}")


# --- helpers --------------------------------------------------------------------

def update_ema(ema, model, decay=0.9995):
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), model.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1 - decay)
        # buffers (e.g. BN running stats in the projector) are copied, not EMA'd —
        # they are already running averages; without this the EMA copy keeps its
        # init stats forever and eval probes are wrong
        for eb, b in zip(ema.buffers(), model.buffers()):
            eb.copy_(b)


def tokens_to_spatial(z):                                   # [B,N,C] -> [B,C,H,W]
    b, n, c = z.shape
    h = int(math.sqrt(n))
    return z.transpose(1, 2).view(b, c, h, h)


def spatial_to_tokens(z):                                   # [B,C,H,W] -> [B,N,C]
    b, c, h, w = z.shape
    return z.view(b, c, h * w).transpose(1, 2)


def sample_t(bs, device, shift=8.0):
    t = torch.sigmoid(torch.randn(bs, device=device))
    return shift * t / (1 + (shift - 1) * t)


def _strip_prefixes(sd):
    out = {}
    for k, v in sd.items():
        for p in ("module.", "_orig_mod."):
            while k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def make_loader(data_dir, image_size, batch_size, num_workers, world_size, rank):
    t = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
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


@torch.no_grad()
def euler_sample(dit, noise, labels, num_steps=50, t_eps=0.05):
    x = noise.clone()
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=x.device)
    for i in range(num_steps):
        t, dt = ts[i], ts[i] - ts[i + 1]
        pred = dit(x, t.expand(x.shape[0]), context=labels, attn_mask=None)
        if isinstance(pred, tuple):
            pred = pred[0]
        v = (x - pred) / max(t.item(), t_eps)
        x = x - dt * v
    return x


# --- Main -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="flat YAML config (configs/e2e/*.yaml)")
    parser.add_argument("--data",          default=None)
    parser.add_argument("--image-size",    type=int, default=256)
    parser.add_argument("--epochs",        type=int, default=10)
    parser.add_argument("--batch-size",    type=int, default=24, help="Per-GPU batch size")
    parser.add_argument("--lr-pd",         type=float, default=1e-4, help="projector+decoder AdamW lr")
    parser.add_argument("--lr-dit",        type=float, default=2e-4, help="DiT gmuon lr")
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--w-rec",         type=float, default=1.0)
    parser.add_argument("--lpips-w",       type=float, default=1.0)
    parser.add_argument("--sigreg-w",      type=float, default=0.02,
                        help="official LeJEPA lambda; pairs with the N-scaled statistic")
    parser.add_argument("--w-fm",          type=float, default=1.0)
    parser.add_argument("--w-pix",         type=float, default=0.0)
    parser.add_argument("--base-coeff",    type=float, default=1.0, help="IG base head FM loss coeff")
    parser.add_argument("--cfg-dropout",   type=float, default=0.1)
    parser.add_argument("--t-shift",       type=float, default=8.0)
    parser.add_argument("--t-eps",         type=float, default=0.05)
    # safety valves (default OFF = pure user design)
    parser.add_argument("--layer-drop",    type=float, default=0.0,
                        help="dropmean-style random layer dropout prob (0 = off, plain mean). "
                             "Train-time only; makes z stochastic -> FM target jitters.")
    parser.add_argument("--detach-fm-target",  action="store_true")
    parser.add_argument("--detach-xt",         action="store_true")
    parser.add_argument("--detach-pix-target", action="store_true")
    parser.add_argument("--pix-t-weight",      action="store_true")
    # init / io
    parser.add_argument("--init-stage1",   type=str, default=None,
                        help="stage-1 ckpt (ema_proj/ema_dec) to warm-start projector+decoder")
    parser.add_argument("--init-dit",      type=str, default=None,
                        help="stage-2 ckpt (ema/model) to warm-start the DiT")
    parser.add_argument("--layers",        type=int, nargs="+", default=RAEV2_LAYERS)
    parser.add_argument("--latent-dim",    type=int, default=1024)
    parser.add_argument("--num-classes",   type=int, default=1000)
    parser.add_argument("--ema-decay",     type=float, default=0.9995)
    parser.add_argument("--clip-grad",     type=float, default=1.0)
    parser.add_argument("--val-image",     type=str, default=None)
    parser.add_argument("--log-every",     type=int, default=50)
    parser.add_argument("--sample-every",  type=int, default=2500)
    parser.add_argument("--ckpt-every",    type=int, default=2)
    parser.add_argument("--out-dir",       default="output/train_e2e_sigreg_dit")
    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--wandb-project", default="raev3")
    parser.add_argument("--wandb-entity",  default="hongyangd")
    parser.add_argument("--num-workers",   type=int, default=4)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--precision",     type=str, default="bf16", choices=["fp32", "bf16"])
    # YAML config provides defaults; explicit CLI flags still override them.
    cfg_ns, _ = parser.parse_known_args()
    if cfg_ns.config:
        import yaml
        ycfg = yaml.safe_load(open(cfg_ns.config)) or {}
        parser.set_defaults(**{k.replace("-", "_"): v for k, v in ycfg.items()})
    args = parser.parse_args()
    if args.data is None:
        parser.error("--data is required (set it in --config or on the CLI)")

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
        print(f"Gradient gates: detach_fm={args.detach_fm_target} detach_xt={args.detach_xt}"
              f" detach_pix={args.detach_pix_target} pix_t_weight={args.pix_t_weight}")

    if args.wandb and is_main:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=f"e2e-sigreg-dit-k{len(args.layers)}-bn"
                        + (f"-drop{args.layer_drop}" if args.layer_drop > 0 else ""),
                   config={**vars(args), "global_batch": args.batch_size * world_size},
                   tags=["e2e", "sigreg", "dit", "stage1+2", f"k{len(args.layers)}"])

    # -- Data ------------------------------------------------------------------
    train_loader, train_sampler = make_loader(
        args.data, args.image_size, args.batch_size, args.num_workers, world_size, rank)
    if is_main:
        print(f"Train: {len(train_loader.dataset)} images")

    # -- Frozen DINOv3 encoder ---------------------------------------------------
    from encoders.vision_encoder import create_encoder
    layers_str = ".".join(str(l) for l in args.layers)
    encoder = create_encoder(f"dinov3mls-vit-l16[layers={layers_str}]",
                             device=device, resolution=args.image_size)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    enc_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    enc_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    def encode_layers(imgs01):
        imgs_norm = (imgs01 - enc_mean) / enc_std
        with torch.no_grad():
            return list(encoder.model.get_intermediate_layers(
                imgs_norm, n=encoder.layer_indices, reshape=False,
                return_class_token=False, norm=True))

    # -- Trainable: projector + decoder + DiT -----------------------------------
    projector = MLSProjector(dim=encoder.hidden_size, out_dim=args.latent_dim,
                             p_drop=args.layer_drop).to(device)

    from stage1.rae import _load_decoder
    decoder = _load_decoder("configs/decoder/ViTXL", hidden_size=args.latent_dim,
                            patch_size=16, num_patches=256, pretrained_path=None).to(device)

    if args.init_stage1:
        ck = torch.load(args.init_stage1, map_location="cpu", weights_only=False)
        projector.load_ln_ckpt(ck["ema_proj"])
        decoder.load_state_dict(ck["ema_dec"])
        if is_main:
            print(f"Warm-started projector+decoder from {args.init_stage1} (ep {ck.get('epoch')})")
        del ck
    elif is_main:
        print("No --init-stage1: projector + decoder trained FROM SCRATCH (pure e2e)")

    dit = DiTwDDTHeadIG(
        input_size=16, patch_size=[1, 1], in_channels=args.latent_dim,
        hidden_size=[1440, 2048], depth=[28, 2], num_heads=[20, 16], mlp_ratio=4.0,
        base_model_depth=8, num_classes=args.num_classes,
        condition_type="label", context_dim=None,
        cond_arch=ConditioningArchConfig(num_t_tokens=4, num_c_tokens=8),
    ).to(device)
    if args.init_dit:
        ck = torch.load(args.init_dit, map_location="cpu", weights_only=False)
        sd = _strip_prefixes(ck.get("ema", ck.get("model", ck)))
        missing, unexpected = dit.load_state_dict(sd, strict=False)
        if is_main:
            print(f"Warm-started DiT from {args.init_dit} (missing {len(missing)}, unexpected {len(unexpected)})")
        del ck, sd

    projector_ddp = DDP(projector, device_ids=[local_rank])
    decoder_ddp   = DDP(decoder,   device_ids=[local_rank])
    dit_ddp       = DDP(dit,       device_ids=[local_rank])

    # EMA only for the DiT (raw-vs-EMA gap in diffusion sampling is large, and the
    # stage-2 baseline FIDs are EMA — keep comparable). Projector/decoder are
    # evaluated LIVE: more responsive collapse alarm, no 1/(1-decay)-step lag.
    ema_dit = deepcopy(dit); ema_dit.requires_grad_(False); ema_dit.eval()

    pd_params  = list(projector.parameters()) + list(decoder.parameters())
    dit_params = list(dit.parameters())
    if is_main:
        print(f"Trainable: projector+decoder {sum(p.numel() for p in pd_params)/1e6:.1f}M"
              f"  |  DiT {sum(p.numel() for p in dit_params)/1e6:.1f}M")

    # -- Optimizers + schedulers -------------------------------------------------
    opt_pd = torch.optim.AdamW(pd_params, lr=args.lr_pd, betas=(0.9, 0.95), weight_decay=0.0)
    opt_dit, msg = build_optimizer(dit_params, OptimizerConfig(
        type="gmuon", lr=args.lr_dit, momentum=0.95, nesterov=True, weight_decay=0.0))
    if is_main:
        print(f"DiT optimizer: {msg}")

    total_steps  = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))

    sched_pd  = torch.optim.lr_scheduler.LambdaLR(opt_pd,  lr_fn)
    sched_dit = torch.optim.lr_scheduler.LambdaLR(opt_dit, lr_fn)

    # -- LPIPS -------------------------------------------------------------------
    from stage1.disc.lpips import LPIPS as LPIPS_
    lpips_all = LPIPS_().to(device).eval()
    for p in lpips_all.parameters():
        p.requires_grad_(False)

    # -- Fixed val images for probes (rank 0) -------------------------------------
    val_layers_fixed, val_img_orig, val_eps, val_labels = None, None, None, None
    if args.val_image and is_main:
        from PIL import Image as PILImage
        _t = transforms.Compose([
            transforms.Resize(args.image_size + 32),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ])
        val_dir = os.path.dirname(os.path.abspath(args.val_image))
        val_paths = sorted(
            p for p in (glob.glob(os.path.join(val_dir, "*.png")) +
                        glob.glob(os.path.join(val_dir, "*.jpg")))
            if "concat" not in os.path.basename(p).lower()) or [args.val_image]
        val_img_orig = torch.stack([_t(PILImage.open(p).convert("RGB")) for p in val_paths]).to(device)
        with torch.no_grad():
            val_layers_fixed = [t.detach() for t in encode_layers(val_img_orig)]
        g = torch.Generator(device=device).manual_seed(args.seed)
        val_eps = torch.randn(val_img_orig.shape[0], args.latent_dim, 16, 16,
                              device=device, generator=g)
        val_labels = torch.full((val_img_orig.shape[0],), args.num_classes, device=device)  # null class
        print(f"Val images pre-encoded: {len(val_paths)} from {val_dir}", flush=True)

    # -- Auto-resume ---------------------------------------------------------------
    start_epoch, global_step = 0, 0
    latest = os.path.join(args.out_dir, "ckpt_latest.pt")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        projector.load_state_dict(ck["projector"]); decoder.load_state_dict(ck["decoder"])
        dit.load_state_dict(ck["dit"])
        ema_dit.load_state_dict(ck["ema_dit"])
        opt_pd.load_state_dict(ck["opt_pd"]); opt_dit.load_state_dict(ck["opt_dit"])
        sched_pd.load_state_dict(ck["sched_pd"]); sched_dit.load_state_dict(ck["sched_dit"])
        start_epoch, global_step = ck["epoch"], ck["global_step"]
        if is_main:
            print(f"Resumed from {latest} (epoch={start_epoch}, step={global_step})")
        del ck

    use_bf16 = (args.precision == "bf16")
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)

    def decode_imgs(dec_module, tokens):
        out = dec_module(tokens, drop_cls_token=False).logits
        m = dec_module.module if isinstance(dec_module, DDP) else dec_module
        return (m.unpatchify(out) * enc_std + enc_mean).clamp(0, 1)

    # -- Training ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        projector_ddp.train(); decoder_ddp.train(); dit_ddp.train()
        t0 = time.time()

        for step, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            layer_tokens = encode_layers(imgs)                  # frozen, no grad

            opt_pd.zero_grad(set_to_none=True)
            opt_dit.zero_grad(set_to_none=True)

            with autocast_ctx:
                # branch 1: representation + reconstruction
                z_tok = projector_ddp(layer_tokens)             # [B,256,1024]
                x_rec = decode_imgs(decoder_ddp, z_tok)
                loss_l1    = F.l1_loss(x_rec, imgs)
                loss_lpips = lpips_all(x_rec * 2 - 1, imgs * 2 - 1).mean()

                # branch 2: generation (flow matching on the LIVE latent)
                z_sp = tokens_to_spatial(z_tok)
                z_xt = z_sp.detach() if args.detach_xt else z_sp
                z_tg = z_sp.detach() if args.detach_fm_target else z_sp
                t = sample_t(z_sp.shape[0], device, args.t_shift)
                te = t.view(-1, 1, 1, 1)
                eps = torch.randn_like(z_sp)
                xt = (1 - te) * z_xt + te * eps
                y_in = torch.where(torch.rand_like(t) < args.cfg_dropout,
                                   torch.full_like(labels, args.num_classes), labels)
                zhat, zhat_base = dit_ddp(xt, t, context=y_in, attn_mask=None)

                w_t = (1.0 / te.clamp_min(args.t_eps) ** 2)
                loss_fm = (w_t * (zhat - z_tg) ** 2).mean() \
                        + args.base_coeff * (w_t * (zhat_base - z_tg) ** 2).mean()

                # OPTIONAL pixel-space FM: decode(zhat) vs decode(z) — NOT GT.
                # Not needed for gradient flow (the live FM target already updates
                # the projector); only adds perceptual reweighting of latent errors,
                # at the cost of a 2nd decoder forward + 2nd LPIPS. Off by default.
                if args.w_pix > 0:
                    x_gen = decode_imgs(decoder_ddp, spatial_to_tokens(zhat))
                    pix_target = x_rec.detach() if args.detach_pix_target else x_rec
                    pix_l1    = F.l1_loss(x_gen, pix_target, reduction="none").mean(dim=(1,2,3))
                    pix_lpips = lpips_all(x_gen * 2 - 1, pix_target * 2 - 1).view(-1)
                    pix = pix_l1 + pix_lpips
                    if args.pix_t_weight:
                        pix = pix * (1 - t)
                    loss_pix = pix.mean()
                else:
                    loss_pix = torch.zeros((), device=device)

            # global SIGReg with the official LeJEPA calibration: ECF means
            # all-reduced differentiably (pooled 8-GPU statistic) and scaled by
            # the total sample count, so under H0 the loss floor is O(1) and
            # sigreg_w (~0.02, paper value) is batch-size independent
            loss_sig = sigreg_loss(z_tok.float().reshape(-1, args.latent_dim),
                                   distributed=True, scale_by_n=True)
            loss_rec = loss_l1 + args.lpips_w * loss_lpips

            loss = (args.w_rec * loss_rec
                    + args.sigreg_w * loss_sig
                    + args.w_fm * loss_fm
                    + args.w_pix * loss_pix)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(pd_params, args.clip_grad)
                torch.nn.utils.clip_grad_norm_(dit_params, args.clip_grad)
            opt_pd.step(); opt_dit.step()
            sched_pd.step(); sched_dit.step()

            update_ema(ema_dit, dit, args.ema_decay)
            global_step += 1

            if is_main and global_step % args.log_every == 0:
                zd = gaussian_diag(z_tok)
                ps = psnr(x_rec, imgs)
                print(f"  ep{epoch+1} s{global_step}"
                      f"  loss={loss.item():.4e}"
                      f"  rec={loss_rec.item():.4e}"
                      f"  psnr={ps:.2f}"
                      f"  sig={loss_sig.item():.4e}"
                      f"  fm={loss_fm.item():.4e}"
                      f"  pix={loss_pix.item():.4e}"
                      f"  z(mu={zd['mean']:.2f} sd={zd['std']:.2f})"
                      f"  vard(mu={zd['var_mean']:.2f} sd={zd['var_disp']:.3f})"
                      f"  lr={opt_pd.param_groups[0]['lr']:.2e}", flush=True)
                if args.wandb:
                    import wandb
                    wandb.log({
                        "train/loss": loss.item(), "train/rec": loss_rec.item(),
                        "train/psnr": ps, "train/sigreg": loss_sig.item(),
                        "train/fm": loss_fm.item(), "train/pix": loss_pix.item(),
                        "train/z_mean": zd["mean"], "train/z_std": zd["std"],
                        "train/z_var_mean": zd["var_mean"], "train/z_var_disp": zd["var_disp"],
                        "train/lr_pd": opt_pd.param_groups[0]["lr"],
                        "train/lr_dit": opt_dit.param_groups[0]["lr"],
                    }, step=global_step)

            # -- sample grid (EMA, Euler 50 steps) -------------------------------
            if is_main and args.sample_every > 0 and global_step % args.sample_every == 0 \
                    and val_eps is not None:
                decoder.eval()
                with torch.no_grad(), autocast_ctx:
                    n = val_eps.shape[0]
                    y_fix = torch.arange(n, device=device) * (args.num_classes // max(1, n))
                    z_gen = euler_sample(ema_dit, val_eps, y_fix, num_steps=50, t_eps=args.t_eps)
                    x_smp = decode_imgs(decoder, spatial_to_tokens(z_gen.float()))
                decoder.train()
                out = os.path.join(args.out_dir, f"samples_s{global_step}.png")
                save_image(x_smp.cpu().float(), out, nrow=n)
                print(f"  Samples -> {out}", flush=True)
                if args.wandb:
                    import wandb
                    wandb.log({"samples/fixed": wandb.Image(out)}, step=global_step)

        if is_main:
            print(f"Epoch {epoch+1}/{args.epochs}  time={time.time()-t0:.0f}s", flush=True)

        # -- per-epoch probes on the LIVE projector/decoder (eval mode: BN uses
        # running stats, layer-drop off) + EMA DiT --------------------------------
        if is_main and val_layers_fixed is not None:
            projector.eval(); decoder.eval()
            with torch.no_grad():
                vz_tok = projector(val_layers_fixed)
                v_rec  = decode_imgs(decoder, vz_tok)
                val_ps = psnr(v_rec, val_img_orig)
                print(f"  Val PSNR (EMA): {val_ps:.2f} dB", flush=True)

                vz_sp = tokens_to_spatial(vz_tok)
                parts = []
                for tv in (0.25, 0.5, 0.75, 0.95):
                    xt_v = (1 - tv) * vz_sp + tv * val_eps
                    tt = torch.full((vz_sp.shape[0],), tv, device=device)
                    with autocast_ctx:
                        pred = ema_dit(xt_v, tt, context=val_labels, attn_mask=None)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    img = decode_imgs(decoder, spatial_to_tokens(pred.float()))
                    parts.append((tv, psnr(img, val_img_orig)))
                grid = "  ".join(f"t{int(tv*100)}={v:.2f}" for tv, v in parts)
                print(f"  Denoise PSNR (EMA): {grid}  ceil={val_ps:.2f} dB", flush=True)
            projector.train(); decoder.train()
            if args.wandb:
                import wandb
                wandb.log({"val/psnr": val_ps,
                           **{f"val/denoise_psnr_t{int(tv*100)}": v for tv, v in parts},
                           "val/denoise_psnr_ceiling": val_ps}, step=global_step)

        # -- checkpoint -------------------------------------------------------------
        if is_main:
            ck = {
                "epoch": epoch + 1, "global_step": global_step,
                "projector": projector.state_dict(), "decoder": decoder.state_dict(),
                "dit": dit.state_dict(),
                "ema_dit": ema_dit.state_dict(),
                "opt_pd": opt_pd.state_dict(), "opt_dit": opt_dit.state_dict(),
                "sched_pd": sched_pd.state_dict(), "sched_dit": sched_dit.state_dict(),
                "layers": args.layers, "args": vars(args),
            }
            torch.save(ck, os.path.join(args.out_dir, "ckpt_latest.pt"))
            if (epoch + 1) % args.ckpt_every == 0:
                p = os.path.join(args.out_dir, f"ckpt_ep{epoch+1:03d}.pt")
                torch.save(ck, p)
                print(f"  Saved {p}", flush=True)
            else:
                print(f"  Saved ckpt_latest.pt (ep{epoch+1})", flush=True)

    if args.wandb and is_main:
        import wandb; wandb.finish()
    dist.destroy_process_group()
    if is_main:
        print("Done.")


if __name__ == "__main__":
    main()
