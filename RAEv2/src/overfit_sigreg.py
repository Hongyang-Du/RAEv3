#!/usr/bin/env python3
"""
Single-image overfitting: learn a SIGReg representation from DINOv3 register tokens.

Architecture:
  DINOv3 (frozen) → register tokens [4, 1024]
       ↓  MLPProjector (learned)
  z_reg [4, 1024]
       ↓  TokenUpsampler  (4 → 256 via cross-attention)
  z_up  [256, 1024]
       ↓  ViT Decoder
  reconstructed image

Loss: L1 + LPIPS  (+  SIGReg when training on multiple images)

Usage:
    cd RAEv2
    python src/overfit_sigreg.py --image ../RAE/assets/parrot.png --steps 2000
"""

import sys, os, math, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image as PILImage


# ─── Multi-GPU gather (keeps gradient) ───────────────────────────────────────

class _GatherLayer(torch.autograd.Function):
    """all_gather with gradient support (grad flows back to local rank only)."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        out = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(out, x.contiguous())
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads):
        x, = ctx.saved_tensors
        return grads[torch.distributed.get_rank()].contiguous()


def all_gather_with_grad(z: torch.Tensor) -> torch.Tensor:
    """Gather z from all GPUs; no-op if not in DDP."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return z
    gathered = _GatherLayer.apply(z)
    return torch.cat(gathered, dim=0)


# ─── SIGReg ─────────────────────────────────────────────────────────────────

def sigreg_loss(z: torch.Tensor, n_proj: int = 64) -> torch.Tensor:
    """
    Sketched Isotropic Gaussian Regularization (simplified).
    Pushes the distribution of z toward N(0,I) via random projections
    and the Epps-Pulley characteristic function test.

    z: [B, N, D] or [B, D] — treated as a batch of D-dim samples.
    Works best with B >> 1.  For single image, use over augmented views.
    """
    if z.ndim == 3:
        z = z.reshape(-1, z.shape[-1])   # [B*N, D]
    z = all_gather_with_grad(z)          # gather across GPUs before statistics
    B, D = z.shape
    if B < 4:
        # Not enough samples for meaningful statistics;
        # fall back to simple moment matching
        mean_loss = z.mean(0).pow(2).mean()
        var_loss  = (z.var(0, unbiased=False) - 1).pow(2).mean()
        return mean_loss + var_loss

    # Random unit projections
    A = F.normalize(torch.randn(D, n_proj, device=z.device), dim=0)  # [D, n_proj]
    proj = z @ A        # [B, n_proj]

    # Characteristic function test: E[exp(i*t*z)] = exp(-t²/2) for N(0,1)
    # Approximate with cosine terms and random t values
    t = torch.randn(n_proj, device=z.device)                # [n_proj]
    cos_emp    = torch.cos(proj * t).mean(0)                # [n_proj]
    cos_gauss  = torch.exp(-0.5 * t ** 2)                   # [n_proj]
    sin_emp    = torch.sin(proj * t).mean(0)
    # For N(0,1), E[sin(tZ)] = 0
    loss = ((cos_emp - cos_gauss) ** 2 + sin_emp ** 2).mean()
    return loss


# ─── Token Upsampler: 4 register tokens → 256 patch positions ────────────────

class TokenUpsampler(nn.Module):
    """
    Perceiver-style: 256 learnable query tokens cross-attend to 4 register tokens.
    Output: [B, 256, dim] — spatial tokens ready for the ViT decoder.
    """
    def __init__(self, n_out: int = 256, dim: int = 1024,
                 n_heads: int = 8, depth: int = 3):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_out, dim) * 0.02)
        self.layers  = nn.ModuleList([
            nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=0.0)
            for _ in range(depth)
        ])
        self.norms_q  = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.norms_kv = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
            for _ in range(depth)
        ])
        self.norms_ff = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])

    def forward(self, kv: torch.Tensor) -> torch.Tensor:
        # kv: [B, 4, dim]
        B = kv.shape[0]
        q = self.queries.expand(B, -1, -1)   # [B, 256, dim]
        for attn, nq, nkv, ffn, nff in zip(
                self.layers, self.norms_q, self.norms_kv, self.ffns, self.norms_ff):
            q_n  = nq(q)
            kv_n = nkv(kv)
            attn_out, _ = attn(q_n, kv_n, kv_n)
            q = q + attn_out
            q = q + ffn(nff(q))
        return q   # [B, 256, dim]


# ─── AttnRes Layer Fusion ─────────────────────────────────────────────────────

class AttnResLayerFusion(nn.Module):
    """
    Attention Residuals (AttnRes) over K layers' CLS tokens.

    Each input is a CLS token from one DINOv3 layer [B, enc_dim].
    Uses the last layer's CLS as query, attends over all K layers' keys,
    produces a single representation z [B, out_dim].

    K=1 (only last layer) → softmax over 1 key = weight 1.0 → plain MLP projection.
    """
    def __init__(self, n_layers: int = 4, enc_dim: int = 1024, out_dim: int = 1024):
        super().__init__()
        self.proj   = nn.Linear(enc_dim, out_dim)   # shared: all layers → out_dim
        self.q_proj = nn.Linear(out_dim, out_dim)   # query from last layer
        self.k_proj = nn.Linear(out_dim, out_dim)   # keys  from all layers
        self.norm   = nn.LayerNorm(out_dim)
        self.scale  = out_dim ** -0.5

    def forward(self, cls_list: list[torch.Tensor]) -> torch.Tensor:
        # cls_list: [CLS_{L-k}, ..., CLS_L], each [B, enc_dim]
        z = torch.stack([self.proj(c) for c in cls_list], dim=1)  # [B, K, out_dim]

        q = self.q_proj(z[:, -1:, :])                             # [B, 1, out_dim]
        k = self.k_proj(z)                                         # [B, K, out_dim]
        attn = torch.softmax(
            (q @ k.transpose(-2, -1)) * self.scale, dim=-1        # [B, 1, K]
        )
        out = (attn @ z).squeeze(1)                                # [B, out_dim]
        return self.norm(out)


# ─── LPIPS (reuse RAEv2's disc if available, else skip) ──────────────────────

def load_lpips(device):
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'RAE', 'src'))
        from disc.lpips import LPIPS
        fn = LPIPS().to(device).eval()
        print("LPIPS loaded.")
        return fn
    except Exception as e:
        print(f"LPIPS not available ({e}), using L1 only.")
        return None


# ─── Main training loop ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",       default="../RAE/assets/parrot.png")
    parser.add_argument("--image-size",  type=int, default=256)
    parser.add_argument("--steps",       type=int, default=3000)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--warmup",      type=int, default=200)
    parser.add_argument("--n-layers",    type=int,   default=4,
                        help="Number of DINOv3 layers to use (last n layers' CLS tokens)")
    parser.add_argument("--sigreg-w",    type=float, default=0.01,
                        help="SIGReg weight (use 0 to disable during single-image test)")
    parser.add_argument("--log-every",   type=int, default=200)
    parser.add_argument("--out-dir",     default="output/overfit_sigreg")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load image ─────────────────────────────────────────────────────────────
    t = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])
    img_orig = t(PILImage.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    print(f"Image: {img_orig.shape}")

    # ── DINOv3 encoder (frozen) ────────────────────────────────────────────────
    from encoders.vision_encoder import create_encoder
    encoder = create_encoder("dinov3-vit-l16", device=device, resolution=args.image_size)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Pre-normalize image for DINOv3
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    img_norm = (img_orig - mean) / std

    # Extract CLS tokens from last n_layers layers (frozen, won't change)
    with torch.no_grad():
        out = encoder.model.get_intermediate_layers(
            img_norm, n=args.n_layers, return_class_token=True
        )
        cls_tokens = [cls.detach() for _, cls in out]   # list of n_layers × [1, 1024]
    print(f"CLS tokens: {len(cls_tokens)} layers × {cls_tokens[0].shape}")

    # ── RAEv2 decoder ──────────────────────────────────────────────────────────
    from omegaconf import OmegaConf
    from stage1.rae import _load_decoder
    s1 = OmegaConf.load("configs/stage2/training/imagenet-dinov3l-k7.yaml").stage_1.params
    decoder = _load_decoder(
        s1.decoder_config_path, hidden_size=1024, patch_size=16,
        num_patches=256, pretrained_path=s1.pretrained_decoder_path,
    ).to(device)
    print(f"Decoder loaded from {s1.pretrained_decoder_path}")

    # ── Learnable modules ──────────────────────────────────────────────────────
    # AttnRes fusion: K CLS tokens → z [1, 1024]
    attn_fusion = AttnResLayerFusion(
        n_layers=args.n_layers, enc_dim=1024, out_dim=1024
    ).to(device)
    # Upsampler: z [1, 1024] → spatial tokens [1, 256, 1024] for decoder
    upsampler   = TokenUpsampler(n_out=256, dim=1024, n_heads=8, depth=3).to(device)

    trainable = list(attn_fusion.parameters()) + list(upsampler.parameters())
    n_params = sum(p.numel() for p in trainable) / 1e6
    print(f"Trainable params: {n_params:.2f}M  (AttnRes fusion + upsampler, decoder frozen)")

    # ── Optim ─────────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95))

    def lr_fn(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    # ── LPIPS ──────────────────────────────────────────────────────────────────
    lpips_fn = load_lpips(device)

    # ── Training loop ──────────────────────────────────────────────────────────
    # Decoder encoder_mean / encoder_std for denorm output
    enc_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    enc_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    losses = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)

        # Forward
        z     = attn_fusion(cls_tokens)     # [1, 1024]  ← AttnRes over K layers
        z_up  = upsampler(z.unsqueeze(1))   # [1, 256, 1024]  ← spatial expand

        # Decode: z_up must be in sequence form [B, N, C]
        dec_out = decoder(z_up, drop_cls_token=False).logits  # [B, N, patch_size²*3]
        x_rec   = decoder.unpatchify(dec_out)                 # [B, 3, H, W]
        # Decoder outputs are in DINOv3-normalized space; convert to [0,1]
        x_rec   = (x_rec * enc_std + enc_mean).clamp(0, 1)

        # Reconstruction loss
        loss_l1   = F.l1_loss(x_rec, img_orig)
        loss_lpips = lpips_fn(x_rec * 2 - 1, img_orig * 2 - 1) if lpips_fn else torch.zeros(1, device=device)
        loss_rec   = loss_l1 + loss_lpips

        # SIGReg on z [1, 1024] — single sample stats are weak for 1 image,
        # meaningful when training on a full dataset (batch >> 1)
        loss_sig = sigreg_loss(z) if args.sigreg_w > 0 else torch.zeros(1, device=device)

        loss = loss_rec + args.sigreg_w * loss_sig
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if (step + 1) % args.log_every == 0:
            avg = np.mean(losses[-args.log_every:])
            lr  = optimizer.param_groups[0]["lr"]
            print(f"  step {step+1:5d}/{args.steps}  loss={avg:.4e}"
                  f"  l1={loss_l1.item():.4e}  lpips={loss_lpips.item():.4e}"
                  f"  sigreg={loss_sig.item():.4e}  lr={lr:.2e}", flush=True)

            # Save comparison image
            pair = torch.cat([img_orig, x_rec], dim=-1).squeeze(0)
            pair_np = (pair.permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
            PILImage.fromarray(pair_np).save(
                os.path.join(args.out_dir, f"step_{step+1:06d}.png"))

    # ── Final image ────────────────────────────────────────────────────────────
    pair = torch.cat([img_orig, x_rec], dim=-1).squeeze(0)
    pair_np = (pair.permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
    PILImage.fromarray(pair_np).save(os.path.join(args.out_dir, "final.png"))

    # ── Loss plot ──────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.semilogy(np.convolve(losses, np.ones(20)/20, mode='valid'))
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("SIGReg Stage-1 Overfit")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.out_dir, "loss.png"), dpi=120, bbox_inches='tight')
    plt.close()

    print(f"\nDone. Final loss: {losses[-1]:.4e}")
    print(f"z stats: mean={z.mean():.3f}  std={z.std():.3f}  (target: 0, 1)")
    with torch.no_grad():
        weights = torch.softmax(
            attn_fusion.q_proj(z.unsqueeze(1)) @
            attn_fusion.k_proj(torch.stack([attn_fusion.proj(c) for c in cls_tokens], dim=1)).transpose(-2,-1)
            * attn_fusion.scale, dim=-1
        ).squeeze()
    print(f"Attention weights over {len(cls_tokens)} layers: {weights.detach().cpu().numpy().round(3)}"
          f"  (last layer = rightmost)")
    print(f"Outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
