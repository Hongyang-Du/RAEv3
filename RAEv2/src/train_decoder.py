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

from configs.stage1_decoder import DecoderConfig, PDropScheduleConfig
from encoders.vision_encoder import create_encoder
from stage1.rae import _load_decoder
from stage1.mask_cond import sample_stratified_masks
from stage1.disc import DinoDiscriminator, hinge_d_loss, vanilla_g_loss, calculate_adaptive_weight
from stage1.disc.diffaug import DiffAug
from stage1.disc.lpips import LPIPS as LPIPS_
from stage1.disc.utils import RandomWindowCrop
from overfit_sigreg import sigreg_loss, gaussian_diag, psnr, val_recon_psnr_npz


# ---- decoder finetune modes (freeze / LoRA), env-driven; no-op unless env set -----
class _LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a trainable low-rank update:
    y = base(x) + scale * B(A(x)). B is zero-init so at step 0 the module == base."""
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling


def _inject_lora(module, targets, r, alpha, skip=(), _prefix=""):
    """Recursively wrap target nn.Linear leaves with LoRA. Subtrees whose dotted path
    matches `skip` (e.g. 'decoder_layers.0') are left untouched -> their output is
    bit-identical (used to preserve h1 = block-0 output that the DiT diffuses in)."""
    n = 0
    for name, child in list(module.named_children()):
        path = f"{_prefix}.{name}" if _prefix else name
        if any(path == s or path.startswith(s + ".") for s in skip):
            continue
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, _LoRALinear(child, r, alpha)); n += 1
        else:
            n += _inject_lora(child, targets, r, alpha, skip, path)
    return n


def apply_finetune_mode(decoder, is_main):
    """Env-driven decoder finetune mode (returns True iff LoRA was injected):
      LORA_R=16 [LORA_ALPHA] [LORA_TARGETS=query,key,value,dense] -> freeze base, train only LoRA
      FREEZE_PREFIXES=decoder_embed  -> freeze decoder params whose name starts with any prefix
    No-ops when neither env is set (normal full training)."""
    lora_r = int(os.environ.get("LORA_R", "0") or "0")
    freeze_prefixes = [s for s in os.environ.get("FREEZE_PREFIXES", "").split(",") if s]
    if lora_r > 0:
        alpha = float(os.environ.get("LORA_ALPHA", str(2 * lora_r)))
        targets = set(t for t in os.environ.get("LORA_TARGETS", "query,key,value,dense").split(",") if t)
        # skip block 0 (+ embed/cls/pos are frozen below) so h1 = block-0 output is UNCHANGED
        # -> the DiT that diffuses h1 stays valid. LORA_SKIP="" to LoRA every block instead.
        skip = [s for s in os.environ.get("LORA_SKIP", "decoder_layers.0").split(",") if s]
        n = _inject_lora(decoder, targets, lora_r, alpha, skip=skip)
        for name, p in decoder.named_parameters():
            p.requires_grad_(".lora_A." in name or ".lora_B." in name)
        if is_main:
            tr = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
            print(f"[finetune] LoRA r={lora_r} alpha={alpha} into {n} Linear(s) targets={sorted(targets)} "
                  f"skip={skip} (h1 preserved); trainable decoder={tr/1e6:.3f}M", flush=True)
        return True
    if freeze_prefixes:
        nf = 0
        for name, p in decoder.named_parameters():
            if any(name.startswith(pre) for pre in freeze_prefixes):
                p.requires_grad_(False); nf += 1
        if is_main:
            print(f"[finetune] froze {nf} decoder param-tensors with prefixes {freeze_prefixes}", flush=True)
    return False


def _remap_lora_state(sd, model):
    """Old (pre-LoRA) state_dict -> wrapped names: a LoRA-wrapped Linear puts its base
    params under an extra '.base.' segment, so remap so base weights still load."""
    keys = set(model.state_dict().keys())
    out = {}
    for k, v in sd.items():
        if k in keys:
            out[k] = v; continue
        head, _, tail = k.rpartition(".")
        cand = f"{head}.base.{tail}"
        out[cand if cand in keys else k] = v
    return out


def update_ema(ema, model, decay=0.9995):
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), model.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1 - decay)
        for eb, b in zip(ema.buffers(), model.buffers()):    # BN running stats: copy
            eb.copy_(b)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, I)), fp32, summed over the latent dim, mean over B*N
    (matches train_decoder_mls_kl.py's legacy full-width KL variant)."""
    mu_f, logvar_f = mu.float(), logvar.float()
    return (-0.5 * (1.0 + logvar_f - mu_f.pow(2) - logvar_f.exp())).sum(-1).mean()


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
    # p_drop-schedule overrides: flip direction / endpoints / shape without editing the YAML
    # (e.g. PDROP_START=0.9 PDROP_END=0.1). If the config has no schedule, any of these enables one.
    if any(os.environ.get(k) for k in ("PDROP_START", "PDROP_END", "PDROP_TYPE", "PDROP_WARMUP_FRAC")):
        if T.p_drop_schedule is None: T.p_drop_schedule = PDropScheduleConfig()
        if os.environ.get("PDROP_START"): T.p_drop_schedule.start = float(os.environ["PDROP_START"])
        if os.environ.get("PDROP_END"): T.p_drop_schedule.end = float(os.environ["PDROP_END"])
        if os.environ.get("PDROP_TYPE"): T.p_drop_schedule.type = os.environ["PDROP_TYPE"]
        if os.environ.get("PDROP_WARMUP_FRAC"): T.p_drop_schedule.warmup_frac = float(os.environ["PDROP_WARMUP_FRAC"])
    # preemption resilience: also save ckpt_latest every N steps (not just per epoch), so
    # a preempted job resumes from a recent step instead of restarting the epoch from zero.
    ckpt_every_steps = int(os.environ.get("CKPT_EVERY_STEPS", "0") or "0")

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

    # -- frozen encoder --------------------------------------------------------
    # MLSCombine/DepthAttnCombine keep `layers` at the top level; CompressedCombine
    # nests the actual fusion combine (and its `layers`) under params.inner.params.
    layers = list(C.params["layers"] if "layers" in C.params
                  else C.params["inner"]["params"]["layers"])
    layers_str = ".".join(str(x) for x in layers)
    # encoder family comes from the config (dinov3mls / siglip2mls / eupemls / ...);
    # falls back to the historical dinov3 hardcode when the config omits it.
    enc_prefix = getattr(cfg, "encoder_name", None) or "dinov3mls-vit-l16"
    enc_prefix = str(enc_prefix).split('[')[0]     # tolerate a stray [layers=...]
    encoder = create_encoder(f"{enc_prefix}[layers={layers_str}]",
                             device=device, resolution=D.image_size)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    _enc_family = enc_prefix.split('-')[0]               # dinov3mls / siglip2mls / eupemls
    _is_siglip = _enc_family.startswith("siglip")
    _m, _s = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if _is_siglip \
             else ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))   # siglip -> [-1,1]
    enc_mean = torch.tensor(_m, device=device).view(1, 3, 1, 1)
    enc_std  = torch.tensor(_s, device=device).view(1, 3, 1, 1)
    if is_main:
        print(f"encoder: {enc_prefix}[layers={layers_str}] (family={_enc_family}, "
              f"dim={encoder.hidden_size}, norm={'siglip0.5' if _is_siglip else 'imagenet'})",
              flush=True)

    def encode_layers(imgs01):
        imgs_norm = (imgs01 - enc_mean) / enc_std
        if _is_siglip:
            # HF SigLIP2 exposes no get_intermediate_layers. hidden_states: tuple len N+1;
            # hs[k]=output of block k-1, hs[N]=post_layernorm(last). post_layernorm the
            # non-final selected blocks to match norm=True (hs[N] is already normed).
            hs = encoder.model(imgs_norm, output_hidden_states=True).hidden_states
            post_ln = encoder.model.vision_model.post_layernorm
            N = encoder._num_hidden_layers
            return [hs[N] if li == N - 1 else post_ln(hs[li + 1])
                    for li in encoder.layer_indices]
        return list(encoder.model.get_intermediate_layers(
            imgs_norm, n=encoder.layer_indices, reshape=False,
            return_class_token=False, norm=True))

    latent_dim = cfg.decoder.latent_dim

    # -- combine (instantiated from config) + decoder (from scratch) -----------
    from utils.model_utils import get_obj_from_str
    combine = get_obj_from_str(C.target)(**C.params).to(device)
    MC = cfg.mask_cond                       # None = unconditional (unchanged path)
    decoder = _load_decoder(cfg.decoder.config_path, hidden_size=latent_dim,
                            patch_size=cfg.decoder.patch_size,
                            num_patches=cfg.decoder.num_patches,
                            pretrained_path=None,
                            mask_cond={"K": len(layers), "d_c": MC.d_c} if MC else None).to(device)
    if MC and is_main:
        n_mc = sum(p.numel() for n, p in decoder.named_parameters()
                   if n.startswith(("mask_embedder.", "null_cond", "ada_")))
        print(f"[mask_cond] d_c={MC.d_c} cond_drop={MC.cond_drop} "
              f"sampler=(full {MC.full_frac:.2f} / uniform {MC.uniform_frac:.2f} / "
              f"bernoulli {1 - MC.full_frac - MC.uniform_frac:.2f})  +{n_mc/1e6:.2f}M params", flush=True)

    # finetune mode (freeze / LoRA) via env; no-op for normal training. MUST run before
    # DDP-wrapping the decoder so frozen params are excluded from DDP grad sync.
    lora_injected = apply_finetune_mode(decoder, is_main)
    if lora_injected:
        # _inject_lora() creates fresh nn.Linear adapters AFTER decoder.to(device) above,
        # so they default to CPU; re-sync the whole tree or DDP() below throws a
        # cpu/cuda device-mismatch ValueError.
        decoder = decoder.to(device)

    # Stage-0 JEPA warm-start: load ONLY the combine from a stage-0 fusion ckpt and
    # FREEZE it (decoder trains from scratch; the recon->combine gradient is severed,
    # which is the whole point of the decoupled-fusion plan). Mask sampling still fires
    # in the loop (combine.train()) since it's gated on self.training, not requires_grad.
    if os.environ.get("STAGE0_COMBINE"):
        _ck0 = torch.load(os.environ["STAGE0_COMBINE"], map_location="cpu", weights_only=False)
        combine.load_state_dict(_ck0["combine"], strict=True)
        combine.requires_grad_(False)
        del _ck0
        if is_main:
            print(f"[stage-0] loaded + FROZE combine from {os.environ['STAGE0_COMBINE']}", flush=True)

    # has_params: does the combine have ANY params (drives train-mode + sigreg below).
    combine_has_params = combine.has_params if hasattr(combine, "has_params") \
        else any(True for _ in combine.parameters())
    # DDP-wrap the combine ONLY if it has TRAINABLE params. A frozen-but-has-params
    # combine (Stage-1 on a frozen JEPA fusion) wrapped in DDP trips the
    # unused-parameter check every step.
    combine_trainable = any(p.requires_grad for p in combine.parameters())
    combine_ddp = DDP(combine, device_ids=[local_rank]) if combine_trainable else combine
    decoder_ddp = DDP(decoder, device_ids=[local_rank])
    ema_combine = deepcopy(combine); ema_combine.requires_grad_(False); ema_combine.eval()
    ema_dec = deepcopy(decoder); ema_dec.requires_grad_(False); ema_dec.eval()

    # -- semantic rent (joint fusion+decoder anti-collapse heads) --------------
    SR = cfg.semantic_rent
    sem_heads = sem_heads_ddp = None
    if SR is not None:
        from stage1.semantic_rent import SemanticRentHeads
        sem_heads = SemanticRentHeads(latent_dim, len(layers), list(SR.mid_layers),
                                      list(SR.deep_layers), hidden=SR.head_hidden,
                                      momentum=SR.momentum).to(device)
        sem_heads_ddp = DDP(sem_heads, device_ids=[local_rank])
        if is_main:
            print(f"[semantic_rent] heads: mid={sem_heads.mid_idx} deep={sem_heads.deep_idx} "
                  f"hidden={SR.head_hidden} | gan_into_fusion={SR.gan_into_fusion} "
                  f"recon_phase={SR.recon_phase} | {sum(p.numel() for p in sem_heads.parameters())/1e6:.2f}M",
                  flush=True)

    trainable = [p for p in list(combine.parameters()) + list(decoder.parameters())
                 + (list(sem_heads.parameters()) if sem_heads is not None else [])
                 if p.requires_grad]
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

    # -- optional random-drop-rate schedule: anneal MLSCombine.p_drop start->end ----
    # progress = global_step/total_steps (micro-batch granularity) so it resumes from a
    # restored global_step. Active only when p_drop_schedule set and combine has p_drop.
    pdrop_sched = T.p_drop_schedule
    pdrop_dynamic = pdrop_sched is not None and hasattr(combine, "p_drop")

    def p_drop_at(step):
        p = step / max(1, total_steps - 1)
        wf = pdrop_sched.warmup_frac
        frac = 0.0 if p < wf else (p - wf) / max(1e-8, 1.0 - wf)
        frac = min(1.0, max(0.0, frac))
        if pdrop_sched.type == "cosine":
            frac = 0.5 * (1.0 - math.cos(math.pi * frac))
        return pdrop_sched.start + (pdrop_sched.end - pdrop_sched.start) * frac

    if is_main and pdrop_dynamic:
        print(f"p_drop schedule: {pdrop_sched.start} -> {pdrop_sched.end} "
              f"({pdrop_sched.type}, warmup_frac={pdrop_sched.warmup_frac})", flush=True)
    elif is_main and pdrop_sched is not None:
        print("WARNING: p_drop_schedule set but combine has no p_drop attr -> ignored", flush=True)

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
    # semantic-rent controller state (survives resume via ckpt["sem_state"])
    sem_state = {"lambda_sem": (SR.lambda_init if SR is not None else 0.0),
                 "phaseB": False, "r_deep_ema": 0.0}
    latest = os.path.join(T.out_dir, "ckpt_latest.pt")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        combine.load_state_dict(ck["combine"]); decoder.load_state_dict(ck["decoder"])
        ema_combine.load_state_dict(ck["ema_combine"]); ema_dec.load_state_dict(ck["ema_dec"])
        disc.load_state_dict(ck["disc"]); optimizer.load_state_dict(ck["optimizer"])
        disc_optimizer.load_state_dict(ck["disc_optimizer"])
        if sem_heads is not None and ck.get("sem_heads") is not None:
            sem_heads.load_state_dict(ck["sem_heads"])
        if ck.get("sem_state") is not None:
            sem_state.update(ck["sem_state"])
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

        # Depth-attn (Variant B) warm-start from a param-free-combine ckpt: the
        # fusion.* params are absent in ck and stay at their zero-init no-op ->
        # step 0 == the anchor's masked mean. Anything else missing is a bug.
        def _load_warm_combine(mod, sd, tag):
            if set(sd.keys()) == set(mod.state_dict().keys()):
                mod.load_state_dict(sd)
                return 0
            _mk, _uk = mod.load_state_dict(sd, strict=False)
            _bad = [k for k in _mk if not k.startswith("fusion.")]
            assert not _bad and not _uk, \
                f"init_from {tag} mismatch beyond fusion modules: missing={_bad} unexpected={list(_uk)}"
            return len(_mk)

        n_fresh_comb = _load_warm_combine(combine, ck["combine"], "combine")
        # LoRA renames base params (extra '.base.'); remap + non-strict so base loads and
        # the fresh LoRA params (absent in ck) stay at init. Plain/freeze -> strict load.
        # mask_cond warm-start from an UNconditional ckpt: also non-strict; the zero-init
        # AdaLN makes the step-0 model function-identical to the loaded ckpt. Fail loudly
        # if anything besides the cond modules is missing.
        _mc_prefixes = ("mask_embedder.", "null_cond", "ada_shared.", "ada_gain", "ada_bias")
        _dec_sd = _remap_lora_state(ck["decoder"], decoder) if lora_injected else ck["decoder"]
        _ema_sd = _remap_lora_state(ck["ema_dec"], ema_dec) if lora_injected else ck["ema_dec"]
        _strict = not (lora_injected or MC)
        mk, uk = decoder.load_state_dict(_dec_sd, strict=_strict)
        ema_dec.load_state_dict(_ema_sd, strict=_strict)
        if MC and not lora_injected:
            bad = [k for k in mk if not k.startswith(_mc_prefixes)]
            assert not bad and not uk, \
                f"init_from mismatch beyond mask_cond modules: missing={bad} unexpected={list(uk)}"
            if is_main:
                print(f"[mask_cond] warm-start: {len(mk)} cond params fresh (zero-init identity)", flush=True)
        _load_warm_combine(ema_combine, ck["ema_combine"], "ema_combine")
        if n_fresh_comb and is_main:
            print(f"[depth_attn] warm-start: {n_fresh_comb} fusion params fresh (zero-init no-op)", flush=True)
        if "disc" in ck:
            disc.load_state_dict(ck["disc"])
        if is_main:
            print(f"Warm-started weights from {T.init_from} (fresh optimizer, epoch=0)"
                  + (f"; LoRA base loaded, missing={len(mk)} unexpected={len(uk)}" if lora_injected else ""))
        del ck

    # SIGReg regularizes the projector's latent -> it only has effect with a
    # parametric projector, and the proven recipe (LeWM) is the BN-MLP (or the new
    # VAE bottleneck). Forbid the silent no-op (projector=none) and the untested ln
    # combo: sigreg => bn | vae.
    if L.sigreg is not None and C.params.get("projector") not in ("bn", "vae"):
        raise ValueError(
            f"loss.sigreg is set but combine.projector={C.params.get('projector')!r}; "
            "SIGReg requires projector: bn or vae. Set one of those or disable sigreg.")
    use_sig = L.sigreg is not None and combine_has_params
    # KL only has effect on the VAE bottleneck's (mu, logvar) head (variational=true);
    # ignored (with a warning) if set on a deterministic combine.
    use_kl = L.kl is not None and getattr(combine, "variational", False)
    if L.kl is not None and not use_kl and is_main:
        print("WARNING: loss.kl is set but combine.variational is not True -> KL ignored", flush=True)
    # align-to-full-sum-mean: the fusion learns to recover the full-layer sum-mean from the
    # random-drop subset (loss below aligns the DROPPED z's fusion part -> full_mean; cls is
    # cancelled in the target so it stays a decoder-side add-on). Needs a trainable combine
    # (recon+align must reach the fusion) AND cls_surrogate=true so z = fusion + cls (decoder
    # eats that) and stage-2 rae.encode() re-adds the same cls -> latent stays consistent.
    use_align = L.align is not None and L.align.weight > 0 and combine_has_params
    if use_align:
        assert combine_trainable, \
            "loss.align is set but the combine is frozen (STAGE0_COMBINE freeze). Align needs " \
            "recon+align gradients to reach the fusion -> load the combine trainable, not frozen."
        assert getattr(combine, "cls_surrogate", False), \
            "loss.align targets (mean+cls); set combine.params.cls_surrogate=true so z carries " \
            "cls (else align would fight the missing cls and stage-2 rae.encode would drop it)."
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(T.precision == "bf16"))

    def decode_imgs(dec, z, layer_mask=None, cond_drop=None):
        out = dec(z, drop_cls_token=False, layer_mask=layer_mask, cond_drop=cond_drop).logits
        m = dec.module if isinstance(dec, DDP) else dec
        return (m.unpatchify(out) * enc_std + enc_mean).clamp(0, 1)

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

    def _mask_for(idx, B):
        """Conditioning mask matching a probe subset (None = full feed). c/z must
        share the mask, so every val decode with a subset idx passes the k-hot."""
        if not MC:
            return None
        m = torch.zeros(B, len(layers), dtype=torch.bool, device=device)
        if idx is None:
            m[:] = True
        else:
            m[:, list(idx)] = True
        return m

    def _async_val_save(sn_combine, sn_dec, ckpt_cpu, ep, gstep, do_probe):
        wb = None
        if cfg.wandb.enabled:
            import wandb as wb
        # -- demo val
        if val_layers_fixed is not None:
            with torch.no_grad():
                vrec = decode_imgs(sn_dec, sn_combine(val_layers_fixed),
                                   layer_mask=_mask_for(None, val_img_orig.shape[0]))
            vps_demo = psnr(vrec, val_img_orig)
            print(f"  [val ep{ep}] PSNR (EMA, {vrec.shape[0]} demo imgs): {vps_demo:.2f} dB", flush=True)
            if wb is not None:
                wb.log({"val/psnr_demo": vps_demo, "val/step": gstep})
        # -- val-N subset (functional SSIM: no Metric.compute() PG sync)
        if val_eval_imgs is not None:
            from torchmetrics.functional.image import structural_similarity_index_measure as _ssim_fn
            psnrs, ssims, psnrs_null = [], [], []
            with torch.no_grad():
                for i in range(0, val_eval_imgs.shape[0], 32):
                    vb = val_eval_imgs[i:i + 32].to(device).float() / 255
                    zv = sn_combine(encode_layers(vb))
                    rec = decode_imgs(sn_dec, zv, layer_mask=_mask_for(None, vb.shape[0])).clamp(0, 1).float()
                    mse = ((rec - vb) ** 2).flatten(1).mean(1)
                    psnrs.append(-10.0 * torch.log10(mse + 1e-10))
                    ssims.append(_ssim_fn(rec, vb, data_range=1.0).item() * rec.shape[0])
                    if MC:   # null-embedding A/B: the net conditioning contribution
                        rec0 = decode_imgs(sn_dec, zv).clamp(0, 1).float()
                        mse0 = ((rec0 - vb) ** 2).flatten(1).mean(1)
                        psnrs_null.append(-10.0 * torch.log10(mse0 + 1e-10))
            vpsnr = torch.cat(psnrs).mean().item()
            vssim = sum(ssims) / val_eval_imgs.shape[0]
            print(f"  [val ep{ep}] PSNR (EMA, {D.val_n} val imgs): {vpsnr:.3f} dB  SSIM={vssim:.4f}", flush=True)
            if MC:
                vpsnr0 = torch.cat(psnrs_null).mean().item()
                print(f"  [val ep{ep}] PSNR null-cond: {vpsnr0:.3f} dB  (cond gain {vpsnr - vpsnr0:+.3f})", flush=True)
                if wb is not None:
                    wb.log({"val/psnr_null": vpsnr0, "val/step": gstep})
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
                _vB = val_img_orig.shape[0]
                def _sub_psnr(idx):
                    return psnr(decode_imgs(sn_dec, sn_combine(val_layers_fixed, idx=idx),
                                            layer_mask=_mask_for(idx, _vB)), val_img_orig)
                full = psnr(decode_imgs(sn_dec, sn_combine(val_layers_fixed),
                                        layer_mask=_mask_for(None, _vB)), val_img_orig)
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
            if pdrop_dynamic:                       # anneal random-drop rate start->end
                combine.p_drop = p_drop_at(global_step)

            if MC:
                # mask conditioning: sample the stratified mask ONCE, use it for both
                # the latent (masked mean) and the decoder conditioning; 10% of rows
                # swap c for the learned null embedding (CFG-style cond dropout).
                m_layers = sample_stratified_masks(
                    imgs.shape[0], len(layers), combine.p_drop,
                    full_frac=MC.full_frac, uniform_frac=MC.uniform_frac, device=device)
                drop_cond = torch.rand(imgs.shape[0], device=device) < MC.cond_drop
            elif SR is not None or use_align:
                # semantic rent / align-to-mean need the mask EXTERNALLY (SR: full/subset R
                # split; align: the dropped z is what we align + we need mask[:,-1] to cancel
                # the mask-gated cls). Use the combine's own stratified-sampler knobs.
                m_layers = sample_stratified_masks(
                    imgs.shape[0], len(layers), combine.p_drop,
                    full_frac=getattr(combine, "full_frac", 1 / 3),
                    uniform_frac=getattr(combine, "uniform_frac", 1 / 3), device=device)
                drop_cond = None
            else:
                m_layers = drop_cond = None

            # per-loss gradient routing into the fusion (semantic-rent joint run):
            #   recon_open -> L1+LPIPS reach fusion; gan_open -> GAN reaches fusion.
            # Legacy (SR None): both open == the old joint-fusion behaviour.
            if SR is not None:
                recon_open = sem_state["phaseB"] if SR.recon_phase == "B" else True
                gan_open = SR.gan_into_fusion and sem_state["phaseB"]
            else:
                recon_open = gan_open = True
            dec_mask = m_layers if MC else None   # decoder conditioning only under mask_cond
            with autocast_ctx:
                z = combine_ddp(layer_tokens, mask=m_layers) if m_layers is not None \
                    else combine_ddp(layer_tokens)
                z_pix = z if recon_open else z.detach()
                x_rec = decode_imgs(decoder_ddp, z_pix, layer_mask=dec_mask, cond_drop=drop_cond)
                loss_l1 = F.l1_loss(x_rec, imgs)
                loss_lpips = lpips_all(x_rec * 2 - 1, imgs * 2 - 1).mean()
            loss_rec = loss_l1 + L.lpips_w * loss_lpips
            loss_sig = (sigreg_loss(z.float().reshape(-1, latent_dim),
                                    distributed=L.sigreg.distributed,
                                    scale_by_n=L.sigreg.scale_by_n)
                        if use_sig else torch.zeros((), device=device))
            loss_kl = (kl_divergence(combine.last_mu, combine.last_logvar)
                       if use_kl else torch.zeros((), device=device))

            # align-to-full-sum-mean: the fusion's SOLE objective -- recover the full-layer
            # sum-mean from the random-drop SUBSET. Align the PRE-cls fusion output
            # (combine.last_fusion, stashed in DepthAttnCombine.forward from the dropped
            # recon-path forward above) to the bare full-layer mean. cls is NOT here -- it is
            # the combine's last step (added after the fusion, into z, for the decoder), so it
            # never enters this loss. Residual is nonzero under drop (fills in the dropped
            # layers) and ~0 only at full feed.
            loss_align = torch.zeros((), device=device)
            if use_align:
                full_mean = torch.stack(layer_tokens, dim=0).mean(0)         # [B,N,d] all-layer sum-mean (no drop, no cls)
                loss_align = (combine.last_fusion.float() - full_mean.float()).pow(2).mean()

            # semantic rent: linear heads read z (grad -> fusion+heads), frozen band targets
            loss_sem = torch.zeros((), device=device)
            sem_metrics = {}
            if SR is not None:
                stk = torch.stack(layer_tokens, dim=0)          # [K,B,N,d] (no_grad, frozen)
                sem_losses, sem_metrics = sem_heads_ddp(z, stk, mask=m_layers)
                loss_sem = SR.w_mid * sem_losses["mid"] + SR.w_deep * sem_losses["deep"]

            loss_gan = torch.zeros(1, device=device)
            if use_gan:
                disc_ddp.eval()
                half = max(1, x_rec.shape[0] // 2)   # half-batch GAN
                # generator fake: reuse x_rec unless recon feeds fusion but GAN must NOT
                # (then a separate half-batch forward on sg(z) keeps GAN out of fusion).
                if SR is not None and recon_open and not gan_open:
                    with autocast_ctx:
                        _mh = dec_mask[:half] if dec_mask is not None else None
                        _dh = drop_cond[:half] if drop_cond is not None else None
                        fake_gen = decode_imgs(decoder_ddp, z.detach()[:half],
                                               layer_mask=_mh, cond_drop=_dh)
                else:
                    fake_gen = x_rec[:half]
                with autocast_ctx:
                    fake_aug = disc_aug.aug(fake_gen * 2 - 1)
                    logits_fake, _ = disc_ddp(fake_aug, None)
                loss_gan = vanilla_g_loss(logits_fake)
                # plain last param is frozen under LoRA/freeze finetune modes (grad-less ->
                # autograd.grad() inside calculate_adaptive_weight throws); use the last
                # TRAINABLE param instead (== plain last param when nothing is frozen).
                last_layer = next(reversed([p for p in decoder_ddp.module.parameters() if p.requires_grad]))
                adp_w = calculate_adaptive_weight(loss_rec, loss_gan, last_layer).clamp(0, 1e4).detach()
            else:
                adp_w = torch.tensor(0.0, device=device)

            sig_w = L.sigreg.weight if use_sig else 0.0
            kl_w = L.kl.weight if use_kl else 0.0
            align_w = L.align.weight if use_align else 0.0
            lam_sem = sem_state["lambda_sem"] if SR is not None else 0.0
            loss = loss_rec + sig_w * loss_sig + kl_w * loss_kl + align_w * loss_align \
                + L.gan.disc_weight * adp_w * loss_gan + lam_sem * loss_sem
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
                    real_aug = disc_aug.aug(imgs[:half] * 2 - 1)
                    fake_aug = disc_aug.aug(x_rec[:half].detach() * 2 - 1)
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

            # ---- semantic-rent: rank-synced R_deep(full) -> phase latch + lambda ----
            # every rank runs this every step (collective + deterministic update) so
            # lambda_sem / phaseB stay identical across ranks.
            if SR is not None:
                rdf = sem_metrics.get("R_deep_full")
                val = rdf.detach().float().reshape(()) if rdf is not None else torch.zeros((), device=device)
                cnt = torch.ones((), device=device) if rdf is not None else torch.zeros((), device=device)
                packed = torch.stack([val, cnt])
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                if packed[1] > 0:
                    r_mean = float(packed[0] / packed[1])
                    sem_state["r_deep_ema"] = (SR.ctrl_ema * sem_state["r_deep_ema"]
                                               + (1 - SR.ctrl_ema) * r_mean)
                if not sem_state["phaseB"]:
                    frac = global_step / max(1, total_steps)
                    if (sem_state["r_deep_ema"] >= SR.r_setpoint and frac >= SR.phase_a_min_frac) \
                            or frac >= SR.phase_a_max_frac:
                        sem_state["phaseB"] = True
                        if is_main:
                            print(f"  [semantic_rent] -> Phase B @ step {global_step} "
                                  f"(R_deep_ema={sem_state['r_deep_ema']:.3f})", flush=True)
                if global_step % SR.ctrl_every == 0:
                    _ls = sem_state["lambda_sem"] * min(1.1, max(0.9,
                            1 + SR.ctrl_eta * (SR.r_setpoint - sem_state["r_deep_ema"])))
                    sem_state["lambda_sem"] = float(min(SR.lambda_max, max(SR.lambda_min, _ls)))

            if is_main and global_step % T.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                zd = gaussian_diag(z); ps = psnr(x_rec, imgs)
                print(f"  ep{epoch+1} s{global_step}  loss={loss.item():.4e}  l1={loss_l1.item():.4e}"
                      f"  lpips={loss_lpips.item():.4e}  psnr={ps:.2f}  sig={loss_sig.item():.4e}"
                      f"  kl={loss_kl.item():.4e}  align={loss_align.item():.4e}"
                      f"  gan={loss_gan.item():.4e}  z(mu={zd['mean']:.2f} sd={zd['std']:.2f})"
                      f"  vard(mu={zd['var_mean']:.2f})  lr={lr:.2e}", flush=True)
                sem_log = {}
                if SR is not None:
                    sem_log = {"train/loss_sem": float(loss_sem), "train/lambda_sem": sem_state["lambda_sem"],
                               "train/phaseB": int(sem_state["phaseB"]),
                               "train/R_deep_ema": sem_state["r_deep_ema"]}
                    for k in ("R_mid_full", "R_mid_sub", "R_deep_full", "R_deep_sub"):
                        if k in sem_metrics:
                            sem_log[f"train/{k}"] = float(sem_metrics[k])
                    print(f"     [rent] lam={sem_state['lambda_sem']:.3f} phaseB={int(sem_state['phaseB'])}"
                          f" R_deep(full={sem_log.get('train/R_deep_full', float('nan')):.3f}"
                          f" sub={sem_log.get('train/R_deep_sub', float('nan')):.3f})"
                          f" R_mid(full={sem_log.get('train/R_mid_full', float('nan')):.3f})"
                          f" loss_sem={float(loss_sem):.4e}", flush=True)
                if cfg.wandb.enabled:
                    import wandb
                    wandb.log({"train/loss": loss.item(), "train/l1": loss_l1.item(),
                               "train/lpips": loss_lpips.item(), "train/psnr": ps,
                               "train/sigreg": loss_sig.item(), "train/kl": loss_kl.item(),
                               "train/align": loss_align.item(), "train/gan_g": loss_gan.item(),
                               "train/z_mean": zd["mean"], "train/z_std": zd["std"],
                               "train/z_var_mean": zd["var_mean"], "train/lr": lr, **sem_log}, step=global_step)

            # -- step-based ckpt_latest (preemption resilience) ------------------
            # epoch=`epoch` (not +1) so resume redoes the current epoch from its start
            # but with the model/optimizer/scheduler/global_step restored (no weight loss).
            if is_main and ckpt_every_steps > 0 and global_step % ckpt_every_steps == 0:
                if step_save_thread is not None:
                    step_save_thread.join()
                _snap = _to_cpu({
                    "epoch": epoch, "global_step": global_step, "layers": layers,
                    "combine": combine.state_dict(), "decoder": decoder.state_dict(),
                    "ema_combine": ema_combine.state_dict(), "ema_dec": ema_dec.state_dict(),
                    "disc": disc.state_dict(), "optimizer": optimizer.state_dict(),
                    "sem_heads": sem_heads.state_dict() if sem_heads is not None else None,
                    "sem_state": dict(sem_state) if SR is not None else None,
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
                "sem_heads": sem_heads.state_dict() if sem_heads is not None else None,
                "sem_state": dict(sem_state) if SR is not None else None,
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
