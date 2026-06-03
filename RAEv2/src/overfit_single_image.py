#!/usr/bin/env python3
"""
RAEv2 version of the single-image DiT width overfitting experiment.

RAEv2 uses DINOv3-L encoder (C=1024) and x-prediction transport.
Theorem 1 from the RAE paper still applies: DiT hidden_size must >= C=1024.

Width sweep: [512, 768, 1024, 1152, 1440] around C=1024.

Usage
-----
Parallel sweep with wandb:
    python src/overfit_single_image.py --sweep --wandb \\
        --rae-config configs/stage2/training/imagenet-dinov3l-k7.yaml \\
        --image ../RAE/assets/parrot.png

Single width (debug):
    CUDA_VISIBLE_DEVICES=0 python src/overfit_single_image.py \\
        --hidden-size 1024 --num-steps 5000 --wandb \\
        --rae-config configs/stage2/training/imagenet-dinov3l-k7.yaml \\
        --image ../RAE/assets/parrot.png
"""

import sys
import os
import math
import argparse
import pickle
import subprocess
import time
import json

import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image as PILImage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2.models.lightningDiT import LightningDiT
from stage2.transport.transport import Transport


# ─── model ────────────────────────────────────────────────────────────────────

def make_dit(hidden_size: int, in_channels: int, input_size: int, depth: int,
             num_classes: int = 1) -> LightningDiT:
    num_heads = max(1, hidden_size // 64)
    while hidden_size % num_heads != 0:
        num_heads -= 1
    model = LightningDiT(
        input_size=input_size,
        patch_size=1,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        num_classes=num_classes,
        condition_type="label",
        cond_arch=_DummyCondArch(),
    )
    model.num_heads = num_heads  # RAEv2 LightningDiT doesn't store this
    return model


class _DummyCondArch:
    """Minimal cond_arch with defaults expected by LightningDiT."""
    num_t_tokens = 4
    num_c_tokens = 8


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
        cfg = OmegaConf.load(args.rae_config)
        rae = instantiate_from_config(cfg.stage_1).to(device)
        rae.eval()
        t = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ])
        x = t(PILImage.open(args.image).convert('RGB')).unsqueeze(0).to(device)
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
    row = torch.cat(imgs, dim=-1).squeeze(0)
    return (row.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


# ─── metrics ──────────────────────────────────────────────────────────────────

class MetricSuite:
    def __init__(self, device):
        self.device = device
        self._lpips = None
        self._ssim  = None

    @property
    def lpips(self):
        if self._lpips is None:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'RAE', 'src'))
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
        lpips_val = self.lpips(img_recon * 2 - 1, img_target * 2 - 1).item()
        psnr_val  = 10 * math.log10(1.0 / max(((img_recon - img_target)**2).mean().item(), 1e-10))
        ssim_val  = self.ssim(img_recon, img_target).item()
        return {"lpips": lpips_val, "psnr": psnr_val, "ssim": ssim_val}


# ─── ODE sampling ─────────────────────────────────────────────────────────────

@torch.no_grad()
def ode_sample(model, transport, z_ref, device, n=1, num_steps=50):
    """Euler ODE: t=1 (noise) → t=0 (clean).  RAEv2 direction."""
    drift = transport.get_drift()
    shift = transport.time_dist_shift
    t_grid = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    t_grid = shift * t_grid / (1 + (shift - 1) * t_grid)

    y = torch.zeros(n, dtype=torch.long, device=device)
    x = torch.randn(n, *z_ref.shape[1:], device=device)
    model.eval()
    for i in range(num_steps):
        h = t_grid[i] - t_grid[i + 1]
        t_b = torch.full((n,), t_grid[i].item(), device=device)
        x = x - h * drift(x, t_b, model, context=y, attn_mask=None)
    model.train()
    return x


# ─── single-width worker ──────────────────────────────────────────────────────

def run_worker(args, hidden_size: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    z_target = torch.load(os.path.join(args.output_dir, "_z_target.pt"), map_location=device)
    _, C, H, W = z_target.shape

    # RAE decoder (optional, for pixel-level metrics)
    rae, img_original, img_target, img_target_np, metrics = None, None, None, None, None
    if args.rae_config and args.image:
        from stage1 import RAE
        from utils.model_utils import instantiate_from_config
        from omegaconf import OmegaConf
        rae = instantiate_from_config(OmegaConf.load(args.rae_config).stage_1).to(device)
        rae.eval()
        t_orig = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ])
        img_original = t_orig(PILImage.open(args.image).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            img_target = rae.decode(z_target).clamp(0, 1)
        img_target_np = (img_target.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        metrics = MetricSuite(device)

    # RAEv2 transport: x-prediction
    time_dist_shift = math.sqrt(math.prod((C, H, W)) / 4096)
    transport = Transport(
        prediction="x",
        time_dist_type="logit-normal_0_1",
        time_dist_shift=time_dist_shift,
    )

    torch.manual_seed(args.seed)
    model = make_dit(hidden_size, in_channels=C, input_size=H,
                     depth=args.depth, num_classes=1).to(device)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[d={hidden_size}] {nparams:.1f}M params  heads={model.num_heads}  C={C}", flush=True)

    # wandb
    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"d={hidden_size}",
            group=f"raev2-overfit-C{C}-depth{args.depth}",
            config=dict(
                hidden_size=hidden_size, depth=args.depth, latent_dim=C,
                num_steps=args.num_steps, lr=args.lr, warmup_steps=args.warmup_steps,
                batch_size=1, seed=args.seed, transport="x-prediction",
                encoder="DINOv3-L-k7",
            ),
            tags=[f"d={hidden_size}", f"C={C}", "width-sweep", "raev2"],
            reinit=True,
        )

    y = torch.zeros(1, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    losses, loss_buf = [], []
    z_tgt_np = z_target.cpu().float().numpy()
    img_dir = os.path.join(args.output_dir, "images", f"d{hidden_size}")
    os.makedirs(img_dir, exist_ok=True)

    model.train()
    for step in range(args.num_steps):
        lr = cosine_lr(step, args.num_steps, args.warmup_steps, args.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.zero_grad(set_to_none=True)

        # RAEv2 training_losses: x-prediction, cfg_dropout=0 for single-image
        loss_dict = transport.training_losses(
            model, z_target,
            model_kwargs=dict(context=y, attn_mask=None),
            model_kwargs_null=dict(context=y, attn_mask=None),
            cfg_dropout_prob=0.0,
        )
        loss = loss_dict['loss'].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        loss_buf.append(loss.item())

        if (step + 1) % args.log_interval == 0:
            avg = float(np.mean(loss_buf)); loss_buf.clear()
            print(f"  [d={hidden_size}] step {step+1}/{args.num_steps}  loss={avg:.4e}  lr={lr:.2e}", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"train/loss": avg, "train/lr": lr}, step=step + 1)

        if (step + 1) % args.val_every == 0:
            _run_validation(
                model, transport, z_target, z_tgt_np,
                rae, img_original, img_target, img_target_np, metrics,
                hidden_size, step + 1, args, device, img_dir,
                n_samples=1, tag="val",
            )

    # final validation
    print(f"[d={hidden_size}] Running final validation ({args.n_val_samples} samples)...", flush=True)
    _run_validation(
        model, transport, z_target, z_tgt_np,
        rae, img_original, img_target, img_target_np, metrics,
        hidden_size, args.num_steps, args, device, img_dir,
        n_samples=args.n_val_samples, tag="final",
        compute_fid_flag=True, n_fid=args.n_fid_samples,
    )

    fl = float(np.mean(losses[-100:]))
    with open(os.path.join(img_dir, "metrics.json"), "w") as f:
        json.dump({"hidden_size": hidden_size, "final_loss": fl,
                   "converged": int(fl < 0.1), "all_losses": losses}, f, indent=2)

    if use_wandb:
        import wandb
        wandb.log({"summary/final_loss": fl, "summary/hidden_size": hidden_size,
                   "summary/converged": int(fl < 0.1)}, step=args.num_steps)
        wandb.finish()

    z_recon = ode_sample(model, transport, z_target, device, n=1)
    result = dict(hidden_size=hidden_size, losses=losses,
                  z_recon=z_recon.cpu(), z_target=z_target.cpu())
    with open(os.path.join(args.output_dir, f"_result_d{hidden_size}.pkl"), 'wb') as f:
        pickle.dump(result, f)
    print(f"[d={hidden_size}] DONE  final_loss={fl:.4e}", flush=True)

    # Concatenate all step PNGs into a progression strip
    step_imgs = sorted(glob.glob(os.path.join(img_dir, "step_*.png")))
    if step_imgs:
        frames = [PILImage.open(p) for p in step_imgs]
        strip = PILImage.new("RGB", (sum(f.width for f in frames), frames[0].height))
        x = 0
        for f in frames:
            strip.paste(f, (x, 0)); x += f.width
        strip_path = os.path.join(img_dir, "progression_strip.png")
        strip.save(strip_path)
        print(f"  [d={hidden_size}] Progression strip → {strip_path}", flush=True)


def _run_validation(
    model, transport, z_target, z_tgt_np,
    rae, img_original, img_target, img_target_np, metrics,
    hidden_size, step, args, device, img_dir,
    n_samples=1, tag="val",
    compute_fid_flag=False, n_fid=50,
):
    use_wandb = args.wandb
    if use_wandb:
        import wandb

    z_samples = ode_sample(model, transport, z_target, device, n=n_samples)
    latent_mse_vals = [float(((z_samples[i:i+1].cpu() - z_target.cpu())**2).mean()) for i in range(n_samples)]

    pca_img = latent_pca_rgb(z_tgt_np, z_samples[0].cpu().float().numpy())
    PILImage.fromarray(pca_img).save(os.path.join(img_dir, f"pca_step_{step:06d}.png"))

    log = {
        f"{tag}/latent_mse":     float(np.mean(latent_mse_vals)),
        f"{tag}/latent_mse_std": float(np.std(latent_mse_vals)) if n_samples > 1 else 0.0,
    }
    if use_wandb:
        log[f"{tag}/latent_pca"] = wandb.Image(pca_img, caption=f"d={hidden_size} step={step}")

    if rae is not None:
        with torch.no_grad():
            imgs_recon = [rae.decode(z_samples[i:i+1]).clamp(0, 1) for i in range(n_samples)]

        lpips_vals, psnr_vals, ssim_vals = [], [], []
        for img_r in imgs_recon:
            m = metrics.compute(img_r, img_target)
            lpips_vals.append(m["lpips"]); psnr_vals.append(m["psnr"]); ssim_vals.append(m["ssim"])

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
                f"{tag}/sample_std": float(torch.cat(imgs_recon).std(dim=0).mean()),
            })

        # panel: original | RAE-recon | DiT-recon×N
        panel_imgs = []
        if img_original is not None: panel_imgs.append(img_original)
        panel_imgs.append(img_target)
        panel_imgs.extend(imgs_recon)
        panel_np = decoded_panel(panel_imgs)
        fname = "final_validation.png" if tag == "final" else f"step_{step:06d}.png"
        local_path = os.path.join(img_dir, fname)
        PILImage.fromarray(panel_np).save(local_path)
        print(f"  [d={hidden_size}] Saved {local_path}", flush=True)

        caption = f"{'original | ' if img_original is not None else ''}RAE-recon | DiT-recon×{n_samples}  d={hidden_size} step={step}"
        if use_wandb:
            log[f"{tag}/decoded_images"] = wandb.Image(panel_np, caption=caption)

        if compute_fid_flag and n_fid >= 8:
            print(f"  [d={hidden_size}] Computing FID({n_fid})...", flush=True)
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'RAE', 'src'))
            from eval.fid import _compute_inception_moments_from_arr, _fid_from_moments
            z_fid = ode_sample(model, transport, z_target, device, n=n_fid)
            with torch.no_grad():
                imgs_fid_np = (rae.decode(z_fid).clamp(0, 1)
                               .permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
            target_tiled = np.tile(img_target_np[None], (n_fid, 1, 1, 1))
            mu_g, sg_g = _compute_inception_moments_from_arr(imgs_fid_np, 32, str(device))
            mu_r, sg_r = _compute_inception_moments_from_arr(target_tiled, 32, str(device))
            fid_val = _fid_from_moments(mu_g, sg_g, mu_r, sg_r)
            log[f"{tag}/fid_{n_fid}"] = fid_val
            print(f"  [d={hidden_size}] FID({n_fid})={fid_val:.2f}", flush=True)

    if use_wandb:
        wandb.log(log, step=step)
    model.train()


# ─── coordinator ─────────────────────────────────────────────────────────────

def run_sweep(args):
    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z_target, _ = get_target_latent(args, device0)
    _, C, H, W = z_target.shape
    n_gpu = torch.cuda.device_count()
    print(f"RAEv2 overfit: C={C}, {H}×{W}, encoder=DINOv3-L")
    print(f"Widths: {args.hidden_sizes}  (threshold C={C})\n")

    z_path = os.path.join(args.output_dir, "_z_target.pt")
    torch.save(z_target.cpu(), z_path)

    procs = []
    for i, hs in enumerate(args.hidden_sizes):
        gpu_id = i % n_gpu if n_gpu > 0 else 0
        cmd = [sys.executable, __file__,
               "--hidden-size",    str(hs),
               "--depth",          str(args.depth),
               "--num-steps",      str(args.num_steps),
               "--lr",             str(args.lr),
               "--warmup-steps",   str(args.warmup_steps),
               "--log-interval",   str(args.log_interval),
               "--val-every",      str(args.val_every),
               "--n-val-samples",  str(args.n_val_samples),
               "--n-fid-samples",  str(args.n_fid_samples),
               "--seed",           str(args.seed),
               "--latent-dim",     str(C),
               "--output-dir",     args.output_dir]
        if args.rae_config: cmd += ["--rae-config", args.rae_config]
        if args.image:      cmd += ["--image", args.image]
        if args.wandb:
            cmd += ["--wandb", "--wandb-project", args.wandb_project,
                    "--wandb-entity", args.wandb_entity]

        print(f"  Launching d={hs} on GPU {gpu_id}", flush=True)
        procs.append((hs, subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)})))
        if i < len(args.hidden_sizes) - 1:
            time.sleep(args.launch_delay)

    all_results = {}
    for hs, p in procs:
        ret = p.wait()
        out_path = os.path.join(args.output_dir, f"_result_d{hs}.pkl")
        if ret != 0:
            print(f"WARNING: d={hs} exited {ret}"); continue
        with open(out_path, 'rb') as f:
            all_results[hs] = pickle.load(f)
        os.remove(out_path)

    if os.path.exists(z_path): os.remove(z_path)
    if not all_results: return

    _plot_all(all_results, C, args.depth, args.output_dir)
    print("\n── Summary ──────────────────────────────────────────────────")
    for hs in sorted(all_results):
        fl  = float(np.mean(all_results[hs]['losses'][-100:]))
        mse = float(((all_results[hs]['z_recon'] - all_results[hs]['z_target'])**2).mean())
        print(f"  d={hs:5d}  ({'≥' if hs>=C else '<'} C={C})  {'✓' if fl<0.1 else '✗'}  loss={fl:.4e}  MSE={mse:.4e}")
    print(f"\nOutputs: {args.output_dir}/")


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
    ax1.set_title(f"RAEv2 Loss Curves  (C={C}, depth={depth})")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    bar_c = ['#2ca02c' if hs>=C else '#d62728' for hs in hidden_sizes]
    ax2.bar([str(hs) for hs in hidden_sizes], [final_losses[hs] for hs in hidden_sizes], color=bar_c)
    thres = next((i for i, hs in enumerate(hidden_sizes) if hs >= C), None)
    if thres and thres > 0:
        ax2.axvline(thres-0.5, color='navy', ls='--', lw=2, label=f"d=C={C}")
        ax2.legend(fontsize=9)
    ax2.set_yscale('log'); ax2.set_xlabel("hidden_size")
    ax2.set_ylabel("Final Loss"); ax2.set_title(f"RAEv2 Width vs Convergence (C={C})")
    ax2.grid(True, alpha=0.3, axis='y')
    plt.suptitle(f"RAEv2 Single-Image Overfitting — DINOv3-L (C={C})", fontsize=13, y=1.01)
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
    plt.suptitle(f"RAEv2 PCA Latent Visualisation (C={C})", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/latent_grid.png", dpi=150, bbox_inches='tight')
    plt.close(); print(f"Saved {output_dir}/latent_grid.png")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAEv2 DiT width sweep (C=1024)")
    parser.add_argument("--rae-config",    type=str,   default=None)
    parser.add_argument("--image",         type=str,   default=None)
    parser.add_argument("--image-size",    type=int,   default=256)
    parser.add_argument("--latent-dim",    type=int,   default=1024)
    parser.add_argument("--hidden-sizes",  type=int,   nargs="+",
                        default=[512, 768, 1024, 1152, 1440])
    parser.add_argument("--hidden-size",   type=int,   default=None)
    parser.add_argument("--depth",         type=int,   default=12)
    parser.add_argument("--num-steps",     type=int,   default=5000)
    parser.add_argument("--lr",            type=float, default=5e-4)
    parser.add_argument("--warmup-steps",  type=int,   default=300)
    parser.add_argument("--log-interval",  type=int,   default=100)
    parser.add_argument("--val-every",     type=int,   default=1000)
    parser.add_argument("--n-val-samples", type=int,   default=8)
    parser.add_argument("--n-fid-samples", type=int,   default=50)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--launch-delay",  type=float, default=10.0)
    parser.add_argument("--sweep",         action="store_true")
    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--wandb-project", type=str,   default="rae")
    parser.add_argument("--wandb-entity",  type=str,   default="hongyangd")
    parser.add_argument("--output-dir",    type=str,   default="overfit_results_v2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.hidden_size is not None and not args.sweep:
        # Worker mode: latent must already exist
        run_worker(args, args.hidden_size)
    else:
        device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        z_target, _ = get_target_latent(args, device0)
        z_path = os.path.join(args.output_dir, "_z_target.pt")
        torch.save(z_target.cpu(), z_path)
        run_sweep(args)


if __name__ == '__main__':
    main()
