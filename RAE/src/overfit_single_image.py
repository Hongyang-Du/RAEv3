#!/usr/bin/env python3
"""
Reproduces Section 4.1 of the RAE paper (arxiv 2510.11690):
"Scaling DiT Width to Match Token Dimensionality"

All widths run in parallel, one GPU each, batch=1.
Each width is a separate W&B run under the same group.

Validation (requires pretrained RAE decoder):
  - Every val_every steps: LPIPS / PSNR / SSIM / latent_MSE + decoded image panel
  - Final: 8-sample grid + FID (50 samples vs target×50)

Usage
-----
Parallel sweep with wandb:
    python src/overfit_single_image.py --sweep --wandb \\
        --rae-config configs/stage2/training/ImageNet256/DiTDH-S_DINOv2-B.yaml \\
        --image assets/pixabay_cat.png

Single width (debug):
    CUDA_VISIBLE_DEVICES=0 python src/overfit_single_image.py \\
        --hidden-size 768 --num-steps 5000 --wandb \\
        --rae-config configs/stage2/training/ImageNet256/DiTDH-S_DINOv2-B.yaml \\
        --image assets/pixabay_cat.png
"""

import sys
import os
import math
import argparse
import pickle
import subprocess
import time

import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2.models.lightningDiT import LightningDiT
from stage2.transport import create_transport, Sampler


# ─── model ────────────────────────────────────────────────────────────────────

def make_dit(hidden_size, in_channels, input_size, depth):
    num_heads = max(1, hidden_size // 64)
    while hidden_size % num_heads != 0:
        num_heads -= 1
    return LightningDiT(
        input_size=input_size, patch_size=1,
        in_channels=in_channels, hidden_size=hidden_size,
        depth=depth, num_heads=num_heads,
        class_dropout_prob=0.0, num_classes=1,
        use_rope=True, use_rmsnorm=True, use_swiglu=True,
        use_qknorm=False, wo_shift=False,
    )


def cosine_lr(step, total, warmup, base_lr, min_lr=None):
    if min_lr is None:
        min_lr = base_lr * 0.05
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * p))


# ─── data ─────────────────────────────────────────────────────────────────────

def get_target_latent(args, device):
    """Return (z_target [1,C,H,W], rae_or_None)."""
    if args.rae_config and args.image:
        from stage1 import RAE
        from utils.model_utils import instantiate_from_config
        from omegaconf import OmegaConf
        from PIL import Image
        cfg = OmegaConf.load(args.rae_config)
        rae = instantiate_from_config(cfg.stage_1).to(device)
        rae.eval()
        t = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ])
        x = t(Image.open(args.image).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            z = rae.encode(x)
        return z, rae
    else:
        C, H, W = args.latent_dim, 16, 16
        torch.manual_seed(args.seed)
        return torch.randn(1, C, H, W, device=device), None


# ─── visualisation ────────────────────────────────────────────────────────────

def latent_pca_rgb(z_ref_np, z_np):
    if z_ref_np.ndim == 4: z_ref_np = z_ref_np.squeeze(0)
    if z_np.ndim == 4:     z_np = z_np.squeeze(0)
    C, H, W = z_ref_np.shape
    ref_flat = z_ref_np.reshape(C, -1).T
    mean = ref_flat.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(ref_flat - mean, full_matrices=False)
    pca3 = Vt[:3]
    def _p(arr):
        f = arr.reshape(C, -1).T - mean
        rgb = f @ pca3.T
        for i in range(3):
            lo, hi = rgb[:, i].min(), rgb[:, i].max()
            rgb[:, i] = (rgb[:, i] - lo) / (hi - lo + 1e-8)
        return (rgb * 255).astype(np.uint8).reshape(H, W, 3)
    return _p(z_np)


def decoded_panel(imgs):
    """
    imgs: list of [1,3,H,W] tensors in [0,1].
    Returns HxW*N uint8 numpy array (side-by-side).
    """
    row = torch.cat(imgs, dim=-1).squeeze(0)          # [3, H, W*N]
    return (row.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


# ─── metrics ──────────────────────────────────────────────────────────────────

class MetricSuite:
    """Lazy-load all eval metrics once, reuse across calls."""
    def __init__(self, device):
        self.device = device
        self._lpips = None
        self._ssim  = None

    @property
    def lpips(self):
        if self._lpips is None:
            from disc.lpips import LPIPS
            self._lpips = LPIPS().to(self.device).eval()
        return self._lpips

    @property
    def ssim(self):
        if self._ssim is None:
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            self._ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        return self._ssim

    @torch.no_grad()
    def compute(self, img_recon, img_target):
        """img_recon, img_target: [1,3,H,W] float in [0,1]. Returns dict."""
        lpips_val = self.lpips(img_recon * 2 - 1, img_target * 2 - 1).item()
        psnr_val  = 10 * math.log10(1.0 / max(((img_recon - img_target)**2).mean().item(), 1e-10))
        ssim_val  = self.ssim(img_recon, img_target).item()
        return {"lpips": lpips_val, "psnr": psnr_val, "ssim": ssim_val}


def compute_fid(gen_imgs_np, target_img_np, n_fid, device):
    """
    gen_imgs_np  : [N,H,W,3] uint8 generated images
    target_img_np: [H,W,3] uint8 target image (will be tiled to N)
    Returns FID score (float).
    NOTE: FID with small N is noisy; use as a directional signal only.
    """
    from eval.fid import _compute_inception_moments_from_arr, _fid_from_moments
    target_tiled = np.tile(target_img_np[None], (n_fid, 1, 1, 1))  # [N,H,W,3]
    mu_gen,  sg_gen  = _compute_inception_moments_from_arr(gen_imgs_np,  batch_size=32, device=device)
    mu_ref,  sg_ref  = _compute_inception_moments_from_arr(target_tiled, batch_size=32, device=device)
    return _fid_from_moments(mu_gen, sg_gen, mu_ref, sg_ref)


# ─── ODE sampling ─────────────────────────────────────────────────────────────

@torch.no_grad()
def ode_sample(model, transport, z_ref, device, n=1):
    """Sample n latents from trained DiT. Returns [n,C,H,W]."""
    sampler = Sampler(transport).sample_ode(sampling_method='euler', num_steps=50)
    y = torch.zeros(n, dtype=torch.long, device=device)
    model.eval()
    z_noise = torch.randn(n, *z_ref.shape[1:], device=device)
    z_recon = sampler(z_noise, model.forward, y=y)[-1]
    model.train()
    return z_recon


# ─── single-width worker ──────────────────────────────────────────────────────

def run_worker(args, hidden_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load shared target latent
    z_target = torch.load(os.path.join(args.output_dir, "_z_target.pt"), map_location=device)
    _, C, H, W = z_target.shape

    # Load RAE decoder if available
    rae = None
    img_original = None   # raw input image tensor [1,3,H,W] in [0,1]
    if args.rae_config and args.image:
        from stage1 import RAE
        from utils.model_utils import instantiate_from_config
        from omegaconf import OmegaConf
        from PIL import Image as PILImage_loader
        rae = instantiate_from_config(OmegaConf.load(args.rae_config).stage_1).to(device)
        rae.eval()
        # original input image (before RAE encoding)
        t_orig = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ])
        img_original = t_orig(PILImage_loader.open(args.image).convert('RGB')).unsqueeze(0).to(device)
        # RAE round-trip: decode(encode(original)) — this is the reconstruction ceiling
        with torch.no_grad():
            img_target = rae.decode(z_target).clamp(0, 1)   # [1,3,H,W]
        img_target_np = (img_target.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        metrics = MetricSuite(device)
    else:
        img_target = img_target_np = metrics = None

    transport = create_transport(
        path_type='Linear', prediction='velocity',
        time_dist_type='logit-normal_0_1',
        time_dist_shift=math.sqrt(math.prod((C, H, W)) / 4096),
    )

    torch.manual_seed(args.seed)
    model = make_dit(hidden_size, in_channels=C, input_size=H, depth=args.depth).to(device)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[d={hidden_size}] {nparams:.1f}M params  heads={model.num_heads}", flush=True)

    # ── wandb init ─────────────────────────────────────────────────────────────
    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"d={hidden_size}",
            group=f"overfit-C{C}-depth{args.depth}",
            config=dict(
                hidden_size=hidden_size, depth=args.depth, latent_dim=C,
                num_steps=args.num_steps, lr=args.lr,
                warmup_steps=args.warmup_steps, batch_size=1,
                seed=args.seed, val_every=args.val_every,
                n_val_samples=args.n_val_samples, n_fid_samples=args.n_fid_samples,
                mode="real_image" if rae else "random_latent",
            ),
            tags=[f"d={hidden_size}", f"C={C}", "width-sweep"],
            reinit=True,
        )

    z_tgt_np = z_target.cpu().float().numpy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    y_single = torch.zeros(1, dtype=torch.long, device=device)
    losses, loss_buf = [], []

    model.train()
    for step in range(args.num_steps):
        lr = cosine_lr(step, args.num_steps, args.warmup_steps, args.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = transport.training_losses(model, z_target, model_kwargs=dict(y=y_single))['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        loss_buf.append(loss.item())

        # ── train logging ──────────────────────────────────────────────────────
        if (step + 1) % args.log_interval == 0:
            avg = float(np.mean(loss_buf)); loss_buf.clear()
            print(f"  [d={hidden_size}] step {step+1}/{args.num_steps}  loss={avg:.4e}  lr={lr:.2e}", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"train/loss": avg, "train/lr": lr}, step=step + 1)

        # ── periodic validation ────────────────────────────────────────────────
        if (step + 1) % args.val_every == 0:
            _run_validation(
                model, transport, z_target, z_tgt_np,
                rae, img_original, img_target, img_target_np, metrics,
                hidden_size, step + 1, args, device,
                n_samples=1, tag="val",
            )

    # ── final validation ───────────────────────────────────────────────────────
    print(f"[d={hidden_size}] Running final validation ({args.n_val_samples} samples)...", flush=True)
    _run_validation(
        model, transport, z_target, z_tgt_np,
        rae, img_original, img_target, img_target_np, metrics,
        hidden_size, args.num_steps, args, device,
        n_samples=args.n_val_samples, tag="final",
        compute_fid_flag=True, n_fid=args.n_fid_samples,
    )

    fl = float(np.mean(losses[-100:]))

    # ── save metrics summary JSON locally ─────────────────────────────────────
    import json
    summary = {"hidden_size": hidden_size, "final_loss": fl, "converged": int(fl < 0.1),
               "all_losses": losses}
    with open(os.path.join(args.output_dir, "images", f"d{hidden_size}", "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if use_wandb:
        import wandb
        wandb.log({
            "summary/final_loss":  fl,
            "summary/hidden_size": hidden_size,
            "summary/converged":   int(fl < 0.1),
        }, step=args.num_steps)
        wandb.finish()

    result = dict(hidden_size=hidden_size, losses=losses,
                  z_recon=ode_sample(model, transport, z_target, device, n=1).cpu(),
                  z_target=z_target.cpu())
    with open(os.path.join(args.output_dir, f"_result_d{hidden_size}.pkl"), 'wb') as f:
        pickle.dump(result, f)
    print(f"[d={hidden_size}] DONE  final_loss={float(np.mean(losses[-100:])):.4e}", flush=True)

    # Concatenate all step PNGs horizontally into one progression strip
    step_imgs = sorted(glob.glob(os.path.join(img_dir, "step_*.png")))
    if step_imgs:
        frames = [PILImage.open(p) for p in step_imgs]
        total_w = sum(f.width for f in frames)
        strip = PILImage.new("RGB", (total_w, frames[0].height))
        x = 0
        for f in frames:
            strip.paste(f, (x, 0))
            x += f.width
        strip_path = os.path.join(img_dir, "progression_strip.png")
        strip.save(strip_path)
        print(f"  [d={hidden_size}] Progression strip → {strip_path}", flush=True)


def _run_validation(
    model, transport, z_target, z_tgt_np,
    rae, img_original, img_target, img_target_np, metrics,
    hidden_size, step, args, device,
    n_samples=1, tag="val",
    compute_fid_flag=False, n_fid=50,
):
    from PIL import Image as PILImage
    use_wandb = args.wandb
    if use_wandb:
        import wandb

    # Local image dir: <output_dir>/images/d<hidden_size>/
    img_dir = os.path.join(args.output_dir, "images", f"d{hidden_size}")
    os.makedirs(img_dir, exist_ok=True)

    # ── latent metrics (always available) ─────────────────────────────────────
    z_samples = ode_sample(model, transport, z_target, device, n=n_samples)
    latent_mse_vals = [float(((z_samples[i:i+1].cpu() - z_target.cpu())**2).mean()) for i in range(n_samples)]
    log = {
        f"{tag}/latent_mse":     float(np.mean(latent_mse_vals)),
        f"{tag}/latent_mse_std": float(np.std(latent_mse_vals)) if n_samples > 1 else 0.0,
    }

    # Save latent PCA image locally
    pca_img = latent_pca_rgb(z_tgt_np, z_samples[0].cpu().float().numpy())
    pca_path = os.path.join(img_dir, f"pca_step_{step:06d}.png")
    PILImage.fromarray(pca_img).save(pca_path)

    if use_wandb:
        log[f"{tag}/latent_pca"] = wandb.Image(
            pca_img, caption=f"d={hidden_size} step={step} (latent PCA)",
        )

    # ── pixel metrics + local save (only if RAE decoder available) ────────────
    if rae is not None:
        with torch.no_grad():
            imgs_recon = [rae.decode(z_samples[i:i+1]).clamp(0, 1) for i in range(n_samples)]

        # per-sample metrics
        lpips_vals, psnr_vals, ssim_vals = [], [], []
        for img_r in imgs_recon:
            m = metrics.compute(img_r, img_target)
            lpips_vals.append(m["lpips"])
            psnr_vals.append(m["psnr"])
            ssim_vals.append(m["ssim"])

        log.update({
            f"{tag}/lpips": float(np.mean(lpips_vals)),
            f"{tag}/psnr":  float(np.mean(psnr_vals)),
            f"{tag}/ssim":  float(np.mean(ssim_vals)),
        })
        if n_samples > 1:
            log.update({
                f"{tag}/lpips_std": float(np.std(lpips_vals)),
                f"{tag}/psnr_std":  float(np.std(psnr_vals)),
                f"{tag}/ssim_std":  float(np.std(ssim_vals)),
            })

        # ── decoded image panel: original | RAE-recon | DiT-recon_0 | ... ───────
        # img_original: raw input pixel image (ground truth)
        # img_target:   decode(z_target) — RAE round-trip reconstruction ceiling
        # imgs_recon:   DiT ODE samples decoded through RAE
        panel_imgs = []
        if img_original is not None:
            panel_imgs.append(img_original)
        panel_imgs.append(img_target)
        panel_imgs.extend(imgs_recon)
        panel_np = decoded_panel(panel_imgs)

        # Save locally — final validation uses its own filename to stay out of the strip
        fname = "final_validation.png" if tag == "final" else f"step_{step:06d}.png"
        local_path = os.path.join(img_dir, fname)
        PILImage.fromarray(panel_np).save(local_path)
        print(f"  [d={hidden_size}] Saved {local_path}", flush=True)

        n_orig = 1 if img_original is not None else 0
        caption = (
            f"{'original | ' if n_orig else ''}RAE-recon | DiT-recon×{n_samples}"
            f"  d={hidden_size} step={step}"
        )
        if use_wandb:
            log[f"{tag}/decoded_images"] = wandb.Image(panel_np, caption=caption)

        # sample variance
        if n_samples > 1:
            stack = torch.cat(imgs_recon, dim=0)
            log[f"{tag}/sample_std"] = float(stack.std(dim=0).mean())

        # FID (final only)
        if compute_fid_flag and n_fid >= 8:
            print(f"  [d={hidden_size}] Computing FID with {n_fid} samples...", flush=True)
            z_fid = ode_sample(model, transport, z_target, device, n=n_fid)
            with torch.no_grad():
                imgs_fid = rae.decode(z_fid).clamp(0, 1)
            imgs_fid_np = (imgs_fid.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
            fid_val = compute_fid(imgs_fid_np, img_target_np, n_fid,
                                  device=str(device))
            log[f"{tag}/fid_{n_fid}"] = fid_val
            print(f"  [d={hidden_size}] FID({n_fid})={fid_val:.2f}", flush=True)

    if use_wandb:
        wandb.log(log, step=step)
    model.train()


# ─── coordinator: parallel launch ────────────────────────────────────────────

def run_sweep(args):
    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z_target, rae = get_target_latent(args, device0)
    _, C, H, W = z_target.shape
    n_gpu = torch.cuda.device_count()
    print(f"Target latent: C={C}, {H}×{W}")
    print(f"Widths: {args.hidden_sizes}  |  {len(args.hidden_sizes)} GPU(s) in parallel\n")

    z_path = os.path.join(args.output_dir, "_z_target.pt")
    torch.save(z_target.cpu(), z_path)

    procs = []
    for i, hs in enumerate(args.hidden_sizes):
        gpu_id = i % n_gpu if n_gpu > 0 else 0
        cmd = [sys.executable, __file__,
               "--hidden-size", str(hs),
               "--depth",        str(args.depth),
               "--num-steps",    str(args.num_steps),
               "--lr",           str(args.lr),
               "--warmup-steps", str(args.warmup_steps),
               "--log-interval", str(args.log_interval),
               "--val-every",    str(args.val_every),
               "--n-val-samples",str(args.n_val_samples),
               "--n-fid-samples",str(args.n_fid_samples),
               "--seed",         str(args.seed),
               "--latent-dim",   str(C),
               "--output-dir",   args.output_dir]
        if args.rae_config: cmd += ["--rae-config", args.rae_config]
        if args.image:      cmd += ["--image", args.image]
        if args.wandb:
            cmd += ["--wandb", "--wandb-project", args.wandb_project,
                    "--wandb-entity", args.wandb_entity]

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        print(f"  Launching d={hs} on GPU {gpu_id}", flush=True)
        procs.append((hs, subprocess.Popen(cmd, env=env)))
        if i < len(args.hidden_sizes) - 1:
            time.sleep(args.launch_delay)

    print(f"\nAll {len(procs)} workers launched. Waiting...\n")
    all_results = {}
    for hs, p in procs:
        ret = p.wait()
        out_path = os.path.join(args.output_dir, f"_result_d{hs}.pkl")
        if ret != 0:
            print(f"WARNING: d={hs} exited {ret}"); continue
        with open(out_path, 'rb') as f:
            all_results[hs] = pickle.load(f)
        os.remove(out_path)
        print(f"  Collected d={hs}", flush=True)

    if os.path.exists(z_path):
        os.remove(z_path)
    if not all_results:
        return

    _plot_all(all_results, C, args.depth, args.output_dir)

    print("\n── Summary ──────────────────────────────────────────────────")
    for hs in sorted(all_results):
        fl  = float(np.mean(all_results[hs]['losses'][-100:]))
        mse = float(((all_results[hs]['z_recon'] - all_results[hs]['z_target'])**2).mean())
        print(f"  d={hs:5d}  ({'≥' if hs>=C else '<'} C={C})  {'✓' if fl<0.1 else '✗'}  loss={fl:.4e}  MSE={mse:.4e}")
    print(f"\nOutputs: {args.output_dir}/")


# ─── plots ────────────────────────────────────────────────────────────────────

def _plot_all(all_results, C, depth, output_dir):
    hidden_sizes = sorted(all_results.keys())
    n = len(hidden_sizes)
    cmap   = plt.cm.RdYlGn
    colors = {hs: cmap(0.1 + 0.8*i/max(n-1,1)) for i, hs in enumerate(hidden_sizes)}
    final_losses = {hs: float(np.mean(all_results[hs]['losses'][-100:])) for hs in hidden_sizes}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for hs in hidden_sizes:
        sm = np.convolve(all_results[hs]['losses'], np.ones(30)/30, mode='valid')
        ax1.semilogy(sm, label=f"d={hs} ({'≥' if hs>=C else '<'}C)", color=colors[hs], lw=1.8)
    ax1.set_xlabel("Steps"); ax1.set_ylabel("Loss (log)")
    ax1.set_title(f"Loss Curves  (C={C}, depth={depth})")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    bar_c = ['#2ca02c' if hs>=C else '#d62728' for hs in hidden_sizes]
    ax2.bar([str(hs) for hs in hidden_sizes], [final_losses[hs] for hs in hidden_sizes], color=bar_c)
    thres = next((i for i, hs in enumerate(hidden_sizes) if hs >= C), None)
    if thres and thres > 0:
        ax2.axvline(thres - 0.5, color='navy', ls='--', lw=2, label=f"d=C={C}")
        ax2.legend(fontsize=9)
    ax2.set_yscale('log'); ax2.set_xlabel("hidden_size")
    ax2.set_ylabel("Final Loss"); ax2.set_title(f"Width vs Convergence (C={C})")
    ax2.grid(True, alpha=0.3, axis='y')
    plt.suptitle(f"RAE §4.1 — Single-Image Overfitting (C={C})", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_curves.png", dpi=150, bbox_inches='tight')
    plt.close(); print(f"Saved {output_dir}/loss_curves.png")

    z_tgt_np = all_results[hidden_sizes[0]]['z_target'].squeeze(0).float().numpy()
    fig2, axes = plt.subplots(1, n+1, figsize=(3*(n+1), 3.5))
    axes[0].imshow(latent_pca_rgb(z_tgt_np, z_tgt_np))
    axes[0].set_title("Target", fontsize=9); axes[0].axis('off')
    for ax, hs in zip(axes[1:], hidden_sizes):
        z_rec_np = all_results[hs]['z_recon'].squeeze(0).float().numpy()
        mse = float(((z_rec_np - z_tgt_np)**2).mean())
        ax.imshow(latent_pca_rgb(z_tgt_np, z_rec_np))
        c = '#2ca02c' if hs>=C else '#d62728'
        ax.set_title(f"d={hs}\n{final_losses[hs]:.1e}\nMSE={mse:.1e}", fontsize=8, color=c)
        ax.axis('off')
    plt.suptitle(f"PCA Latent Visualisation (C={C})", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/latent_grid.png", dpi=150, bbox_inches='tight')
    plt.close(); print(f"Saved {output_dir}/latent_grid.png")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rae-config",    type=str,   default=None)
    parser.add_argument("--image",         type=str,   default=None)
    parser.add_argument("--image-size",    type=int,   default=256)
    parser.add_argument("--latent-dim",    type=int,   default=768)
    parser.add_argument("--hidden-sizes",  type=int,   nargs="+",
                        default=[192, 384, 576, 768, 960, 1152])
    parser.add_argument("--hidden-size",   type=int,   default=None)
    parser.add_argument("--depth",         type=int,   default=12)
    parser.add_argument("--num-steps",     type=int,   default=5000)
    parser.add_argument("--lr",            type=float, default=5e-4)
    parser.add_argument("--warmup-steps",  type=int,   default=300)
    parser.add_argument("--log-interval",  type=int,   default=100)
    parser.add_argument("--val-every",     type=int,   default=1000,
                        help="Validation interval in steps")
    parser.add_argument("--n-val-samples", type=int,   default=8,
                        help="ODE samples for periodic validation")
    parser.add_argument("--n-fid-samples", type=int,   default=50,
                        help="ODE samples for final FID computation")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--launch-delay",  type=float, default=10.0)
    parser.add_argument("--sweep",         action="store_true")
    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--wandb-project", type=str,   default="rae")
    parser.add_argument("--wandb-entity",  type=str,   default="hongyangd")
    parser.add_argument("--output-dir",    type=str,   default="overfit_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.hidden_size is not None and not args.sweep:
        run_worker(args, args.hidden_size)
    else:
        # Need to create the shared latent before spawning workers
        device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        z_target, _ = get_target_latent(args, device0)
        z_path = os.path.join(args.output_dir, "_z_target.pt")
        torch.save(z_target.cpu(), z_path)
        run_sweep(args)


if __name__ == '__main__':
    main()
