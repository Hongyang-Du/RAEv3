"""Latent-geometry spectrum analysis for the h1 ("move the diffusion boundary INTO the
decoder") claim. Computes the per-channel-standardized covariance (== correlation-matrix)
eigenspectrum of the tensors the DiT actually diffuses. A flatter spectrum == a better-
conditioned optimization target, which directly explains the DiT convergence speed-up.

Four featured targets. ALL are standardized per-channel by THIS script, uniformly, so the
comparison is about manifold GEOMETRY, not each run's own normalization scheme (some h1
configs ship per-position stats, z ships per-channel -> not comparable as-is). Note: for a
per-channel-normalized DiT this correlation-matrix spectrum IS exactly the covariance
spectrum of the tensor the DiT optimizes.

  z             shared encoder combine   [., 1024, 16,16]  param-free mean -> decoder-independent
  h1_ours       our decoder, after block 0            [., 1152, 16,16]
  h1_raev2      RAEv2 decoder, after block 0          [., 1152, 16,16]
  h{deep}_ours  our decoder, after a deeper block     [., 1152, 16,16]   (--deep-block)

Because the combine is param-free (weighting=mean, projector=none), z is computed BEFORE the
decoder and is identical for both models within a matched layer set (k7<->k7). We therefore
take z once (from --ours-config) as the single shared curve. h1_ours vs h1_raev2 is then a
clean same-input / same-arch control isolating the decoder-training effect.

Bonus figure: effective-rank / C vs decoder depth for BOTH decoders (all 28 blocks) -> shows
WHERE the spectrum is flattest, i.e. which block is the best diffusion boundary (answers the
reviewer's "why h1 and not deeper?").

Run (single GPU is fine; supports DDP via all-reduce):
  torchrun --nproc_per_node=1 src/eval_latent_spectrum.py \
    --ours-config  configs/stage2/training/imagenet-dinov3l-h1decoder-plain-cls-k7.yaml \
    --raev2-config configs/stage2/training/imagenet-dinov3l-h1decoder-raev2k7.yaml \
    --num-samples 50000 --batch 128 --deep-block -1 \
    --out-dir output_full/latent_spectrum_k7
"""
import argparse
import dataclasses
import json
import os

import torch
from omegaconf import OmegaConf

from configs.stage2 import Stage2Config
from data import prepare_unified_dataloader
from utils.dist_utils import setup_distributed
from utils.model_utils import instantiate_from_config


def load_stage1(config_path, device, ckpt=None):
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(config_path)))
    if ckpt:                                              # override stale ckpt path
        config.stage_1.params["stage1_ckpt_path"] = ckpt
    config.post_process()
    model = instantiate_from_config(config.stage_1).to(device).eval()
    return model, config


@torch.no_grad()
def decoder_hidden_states(model, images):
    """List of per-block patch-token hidden states, each [B, N, Cdec]. Mirrors
    RAEDecoderH1._block0 exactly, then runs ALL decoder blocks (CLS dropped per block)."""
    z = model._combine_tokens(images)                     # [B, N, 1024] (raw shared combine)
    d = model.decoder
    x = d.decoder_embed(z)                                # [B, N, Cdec]
    x = d.interpolate_latent(x)
    cls = d.trainable_cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat([cls, x], dim=1)                        # [B, N+1, Cdec]
    x = x + d.decoder_pos_embed
    outs = []
    for layer in d.decoder_layers:
        x = layer(x, head_mask=None)[0]
        outs.append(x[:, 1:, :])                          # drop CLS -> [B, N, Cdec]
    return z, outs


class Gram:
    """Streaming per-channel sum + Gram (X^T X) over token samples, float64."""

    def __init__(self, C, device):
        self.s = torch.zeros(C, dtype=torch.float64, device=device)
        self.G = torch.zeros(C, C, dtype=torch.float64, device=device)
        self.n = 0

    def update(self, tokens):                             # tokens [., ., C] or [., C]
        X = tokens.reshape(-1, tokens.shape[-1]).double()
        self.s += X.sum(0)
        self.G += X.t() @ X
        self.n += X.shape[0]

    def reduce(self, world_size):
        if world_size > 1:
            torch.distributed.all_reduce(self.s)
            torch.distributed.all_reduce(self.G)
            t = torch.tensor([float(self.n)], dtype=torch.float64, device=self.s.device)
            torch.distributed.all_reduce(t)
            self.n = int(t.item())

    def eigenspectrum(self):
        """Descending eigenvalues of the per-channel CORRELATION matrix (trace == C)."""
        mean = self.s / self.n
        cov = self.G / self.n - torch.outer(mean, mean)
        std = cov.diag().clamp_min(1e-12).sqrt()
        corr = cov / torch.outer(std, std)
        corr = 0.5 * (corr + corr.t())                    # symmetrize away fp error
        eig = torch.linalg.eigvalsh(corr).flip(0).clamp_min(0.0)
        return eig.cpu()


def metrics(eig):
    eig = eig.double()
    total = eig.sum()
    p = (eig / total).clamp_min(1e-20)
    C = eig.numel()
    eff = float(torch.exp(-(p * p.log()).sum()))
    part = float(total * total / (eig * eig).sum())
    stable = float(total / eig[0])
    pos = eig[eig > 1e-12]
    cond = float(eig[0] / pos[-1]) if pos.numel() else float("inf")
    csum = torch.cumsum(eig, 0) / total
    frac = lambda th: (int((csum < th).sum()) + 1) / C
    return {
        "C": C,
        "effective_rank": eff,
        "effective_rank_over_C": eff / C,
        "participation_ratio": part,
        "participation_ratio_over_C": part / C,
        "stable_rank": stable,
        "condition_number": cond,
        "frac_comps_90pct_energy": frac(0.90),
        "frac_comps_99pct_energy": frac(0.99),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours-config", required=True)
    ap.add_argument("--raev2-config", required=True)
    ap.add_argument("--num-samples", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--deep-block", type=int, default=-1,
                    help="featured deeper block index (0=h1); -1 = last block")
    ap.add_argument("--data-dir", default=None, help="override dataset.data_dir (both models)")
    ap.add_argument("--split", default=None, help="override dataset.split (e.g. train/val)")
    ap.add_argument("--ours-ckpt", default=None, help="override ours stage1_ckpt_path")
    ap.add_argument("--raev2-ckpt", default=None, help="override raev2 stage1_ckpt_path")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rank, world_size, device = setup_distributed()
    ours, ours_cfg = load_stage1(args.ours_config, device, ckpt=args.ours_ckpt)
    raev2, _ = load_stage1(args.raev2_config, device, ckpt=args.raev2_ckpt)
    if args.data_dir:
        ours_cfg.dataset.data_dir = args.data_dir
    if args.split:
        ours_cfg.dataset.split = args.split

    n_blocks = len(ours.decoder.decoder_layers)
    deep = args.deep_block if args.deep_block >= 0 else n_blocks - 1
    if rank == 0:
        print(f"decoder blocks: {n_blocks} | featured deep block: {deep} "
              f"(h1 = block 0)", flush=True)

    loader = prepare_unified_dataloader(
        config=dataclasses.asdict(ours_cfg.dataset),
        image_size=ours_cfg.training.image_size,
        batch_size=args.batch,
        num_workers=ours_cfg.training.num_workers,
        rank=rank, world_size=world_size,
        transform=None, condition_type=ours_cfg.conditioning.type,
        virtual_epoch_steps=None,
    )

    gram_z = None
    gram_ours = None
    gram_raev2 = None
    seen = 0
    for images, _ in loader:
        images = images.to(device)
        z, hs_ours = decoder_hidden_states(ours, images)
        _, hs_raev2 = decoder_hidden_states(raev2, images)
        if gram_z is None:
            gram_z = Gram(z.shape[-1], device)
            gram_ours = [Gram(h.shape[-1], device) for h in hs_ours]
            gram_raev2 = [Gram(h.shape[-1], device) for h in hs_raev2]
        gram_z.update(z)
        for g, h in zip(gram_ours, hs_ours):
            g.update(h)
        for g, h in zip(gram_raev2, hs_raev2):
            g.update(h)
        seen += images.shape[0]
        if rank == 0 and seen % (args.batch * 20) == 0:
            print(f"  {seen}/{args.num_samples}", flush=True)
        if seen >= args.num_samples:
            break

    for g in [gram_z, *gram_ours, *gram_raev2]:
        g.reduce(world_size)
    if rank != 0:
        return

    eig_z = gram_z.eigenspectrum()
    eig_ours = [g.eigenspectrum() for g in gram_ours]
    eig_raev2 = [g.eigenspectrum() for g in gram_raev2]

    out = {
        "z": metrics(eig_z),
        "ours": {f"block_{i}": metrics(e) for i, e in enumerate(eig_ours)},
        "raev2": {f"block_{i}": metrics(e) for i, e in enumerate(eig_raev2)},
        "featured": {"deep_block": deep, "n_blocks": n_blocks},
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- compact table for the 4 featured targets ----
    featured = [
        ("z (combine)", eig_z),
        ("h1_ours (block 0)", eig_ours[0]),
        ("h1_raev2 (block 0)", eig_raev2[0]),
        (f"h_ours (block {deep})", eig_ours[deep]),
    ]
    print(f"\n{'target':22s} {'C':>5s} {'effRank/C':>10s} {'PR/C':>7s} "
          f"{'stableRank':>11s} {'cond':>9s} {'90%':>6s} {'99%':>6s}")
    for name, e in featured:
        m = metrics(e)
        print(f"{name:22s} {m['C']:5d} {m['effective_rank_over_C']:10.3f} "
              f"{m['participation_ratio_over_C']:7.3f} {m['stable_rank']:11.1f} "
              f"{m['condition_number']:9.1f} {m['frac_comps_90pct_energy']:6.2f} "
              f"{m['frac_comps_99pct_energy']:6.2f}")

    # ---- figures (quick diagnostic; polish for the paper once the trend is confirmed) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fig 1: featured eigenspectra (rank fraction vs eigenvalue, log-y). Flat at 1 = isotropic.
    plt.figure(figsize=(6, 4.2))
    for name, e in featured:
        xs = (torch.arange(e.numel()) + 1).double() / e.numel()
        plt.semilogy(xs.numpy(), e.numpy(), label=name, lw=1.8)
    plt.axhline(1.0, ls="--", c="gray", lw=1, label="isotropic (eig=1)")
    plt.xlabel("rank fraction  i / C")
    plt.ylabel("correlation eigenvalue")
    plt.title("Latent spectrum (per-channel standardized)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "spectrum_featured.png"), dpi=160)
    plt.close()

    # fig 2: effective-rank/C vs decoder depth, both decoders. Higher = flatter = easier for DiT.
    plt.figure(figsize=(6, 4.2))
    depth = list(range(n_blocks))
    plt.plot(depth, [metrics(e)["effective_rank_over_C"] for e in eig_ours],
             "-o", ms=3, label="ours")
    plt.plot(depth, [metrics(e)["effective_rank_over_C"] for e in eig_raev2],
             "-s", ms=3, label="raev2")
    plt.axhline(out["z"]["effective_rank_over_C"], ls=":", c="k", lw=1,
                label="z (combine)")
    plt.axhline(1.0, ls="--", c="gray", lw=1, label="isotropic")
    plt.axvline(deep, ls="-", c="r", lw=0.8, alpha=0.5)
    plt.axvline(0, ls="-", c="b", lw=0.8, alpha=0.5)
    plt.xlabel("decoder block index  (h1 = block 0)")
    plt.ylabel("effective rank / C")
    plt.title("Spectrum flatness vs decoder depth")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "effrank_vs_depth.png"), dpi=160)
    plt.close()

    print(f"\nsaved -> {args.out_dir}/  (metrics.json, spectrum_featured.png, "
          f"effrank_vs_depth.png)")


if __name__ == "__main__":
    main()
