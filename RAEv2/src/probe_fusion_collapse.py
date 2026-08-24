#!/usr/bin/env python3
"""Collapse readout for the DepthAttnCombine (softmax) fusion network.

Question this answers: does the fusion's softmax depth-attention still collapse onto
the SHALLOWEST layer (index 0 == DINOv3 layer 1 eating ~all the weight), the failure
mode the semantic rent was meant to prevent?

Two complementary, decoder-free measures on real ImageNet-val images:

1) ATTENTION SIMPLEX (the literal "softmax puts all weight on layer 1"):
   each DepthAttnBlock returns its softmax weights w [B, N, K] over the K=23 layers.
   We report, per block:
     - mean weight per layer  (averaged over B*N tokens)          -> the [K] profile
     - normalized entropy  H(w)/log(K)   (1.0 = uniform, ~0 = collapsed to one layer)
        both for the mean profile AND averaged per-token (per-token catches collapse
        that averaging would hide)
     - weight mass on the shallowest layer (index 0) and the argmax-layer histogram
       (what fraction of tokens pick each layer as their top layer)

2) FUNCTIONAL LATENT RELIANCE (leave-one-out on the actual latent, no decoder):
   for each layer k, drop it (combine(idx = all-but-k)) and measure the relative
   latent shift ||z_full - z_LOO|| / ||z_full||. A spike at the shallow end = the
   latent genuinely leans on the shallow layer; a flat/thin spread = healthy fusion.
   Also reports ||correction|| / ||z0|| = how much the learned attention moves the
   latent off the plain equal-mean floor z0 at all.

Single GPU, no DDP. Uses the EMA fusion (ema_combine) by default, matching the DiT
latent. Example:
    python src/probe_fusion_collapse.py \
        --ckpt /sensei-fs-3/users/hongyangd/ckpt/rent-k23-depthattn-softmax-ganfusion-2node/ckpt_latest.pt \
        --npz  /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2/data_eval/imagenet-256-val.npz \
        --n 512
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from encoders.vision_encoder import create_encoder
from stage1.combine import DepthAttnCombine

# fusion params copied verbatim from the training log of
# rent-k23-depthattn-softmax-ganfusion-2node (line "combine: stage1.combine.DepthAttnCombine ...")
FUSION_PARAMS = dict(
    layers=list(range(1, 24)), p_drop=0.5, full_frac=0.3333333, uniform_frac=0.3333333,
    cls_surrogate=True, dim=1024, out_dim=1024, n_layers=2, n_heads=8, mlp_mult=2,
    attn_kind="softmax",
)
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/sensei-fs-3/users/hongyangd/ckpt/"
                    "rent-k23-depthattn-softmax-ganfusion-2node/ckpt_latest.pt")
    ap.add_argument("--npz", default="/sensei-fs-3/users/hongyangd/RAEv3_oldnorm/"
                    "RAEv2/data_eval/imagenet-256-val.npz")
    ap.add_argument("--key", choices=["ema_combine", "combine"], default="ema_combine",
                    help="which fusion state dict to probe (ema = the DiT latent)")
    ap.add_argument("--n", type=int, default=512, help="number of val images")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-loo", action="store_true", help="skip the leave-one-out latent scan")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers_str = ".".join(str(i) for i in FUSION_PARAMS["layers"])

    # -- frozen DINOv3 encoder (same as training's encode_layers) --------------
    encoder = create_encoder(f"dinov3mls-vit-l16[layers={layers_str}]",
                             device=device, resolution=args.resolution)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # -- fusion, loaded strict from the ckpt -----------------------------------
    fusion = DepthAttnCombine(**FUSION_PARAMS).to(device).eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    miss, unexp = fusion.load_state_dict(ck[args.key], strict=True)
    K = fusion.K
    print(f"loaded fusion[{args.key}] from {args.ckpt} (stage-1 epoch {ck.get('epoch')}) "
          f"| missing={list(miss)} unexpected={list(unexp)}")
    if ck.get("sem_state") is not None:
        print(f"ckpt sem_state (rent controller): {ck['sem_state']}")

    imgs_np = np.load(args.npz, mmap_mode="r")["images"]
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(imgs_np.shape[0], generator=g)[:args.n].tolist()
    im_mean = IMAGENET_MEAN.to(device)
    im_std = IMAGENET_STD.to(device)

    n_blocks = FUSION_PARAMS["n_layers"]
    # accumulators
    w_sum = [torch.zeros(K, dtype=torch.float64, device=device) for _ in range(n_blocks)]
    tok_ent_sum = [0.0 for _ in range(n_blocks)]              # sum of per-token norm-entropy
    argmax_hist = [torch.zeros(K, dtype=torch.float64, device=device) for _ in range(n_blocks)]
    n_tokens = 0
    corr_ratio_sum = 0.0
    loo_shift_sum = torch.zeros(K, dtype=torch.float64, device=device)
    n_imgs = 0

    @torch.no_grad()
    def encode_layers(imgs01):
        x = imgs01
        if x.shape[-1] != args.resolution:
            x = F.interpolate(x, size=(args.resolution, args.resolution),
                              mode="bicubic", align_corners=False)
        x = (x - im_mean) / im_std
        return list(encoder.model.get_intermediate_layers(
            x, n=encoder.layer_indices, reshape=False,
            return_class_token=False, norm=True))

    for i in range(0, len(idx), args.batch):
        bidx = idx[i:i + args.batch]
        imgs = torch.from_numpy(np.ascontiguousarray(imgs_np[bidx])).to(device)  # [b,H,W,3] u8
        imgs = imgs.permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            lt = encode_layers(imgs)                          # K x [b, N, 1024]
            z, attns = fusion(lt, return_attn=True)           # z [b,N,1024]; attns: n_blocks x [b,N,K]
            b, N, _ = z.shape
            stk = torch.stack(lt, 0)                          # [K,b,N,1024]
            z0 = stk.mean(0)                                  # plain equal-mean floor (full feed)
            # learned correction magnitude relative to the equal-mean floor
            corr = z - z0
            corr_ratio_sum += (corr.norm(dim=-1) / z0.norm(dim=-1).clamp_min(1e-6)).sum().item()

            for bi in range(n_blocks):
                w = attns[bi].reshape(-1, K).double()         # [b*N, K] softmax simplex
                w_sum[bi] += w.sum(0)
                ent = -(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(1) / np.log(K)  # [b*N]
                tok_ent_sum[bi] += ent.sum().item()
                argmax_hist[bi] += torch.bincount(w.argmax(1), minlength=K).double()
            n_tokens += b * N

            if not args.no_loo:
                allset = list(range(K))
                zf = z.reshape(b, -1)
                for k in range(K):
                    sub = allset[:k] + allset[k + 1:]
                    zk = fusion(lt, idx=sub).reshape(b, -1)
                    loo_shift_sum[k] += ((zf - zk).norm(dim=1)
                                         / zf.norm(dim=1).clamp_min(1e-6)).double().sum()
            n_imgs += b
        print(f"  processed {n_imgs}/{len(idx)}", end="\r", flush=True)
    print()

    log_unif = float(np.log(K))
    print("\n" + "=" * 78)
    print(f"FUSION COLLAPSE READOUT  |  {n_imgs} images, {n_tokens} tokens, K={K} layers "
          f"(index 0 = DINOv3 layer 1 = SHALLOWEST)")
    print("=" * 78)
    for bi in range(n_blocks):
        w_mean = (w_sum[bi] / n_tokens)                       # [K] mean softmax weight
        H_mean = float(-(w_mean.clamp_min(1e-12) * w_mean.clamp_min(1e-12).log()).sum() / log_unif)
        H_tok = tok_ent_sum[bi] / n_tokens                    # mean per-token norm-entropy
        am = (argmax_hist[bi] / n_tokens)                     # [K] argmax share
        top = torch.topk(w_mean, min(5, K))
        print(f"\n--- DepthAttnBlock {bi} (softmax over {K} layers) ---")
        print(f"  norm-entropy: per-token={H_tok:.3f}  mean-profile={H_mean:.3f}   "
              f"(1.000 = uniform 1/{K};  ~0 = collapsed to one layer)")
        print(f"  weight on SHALLOWEST layer (idx0/L1): {w_mean[0].item()*100:.2f}%   "
              f"(uniform would be {100.0/K:.2f}%)")
        print(f"  argmax-share of shallowest layer     : {am[0].item()*100:.2f}%  "
              f"of tokens pick L1 as their top layer")
        print("  top-5 layers by mean weight: " + ", ".join(
            f"L{FUSION_PARAMS['layers'][j]}={w_mean[j].item()*100:.1f}%" for j in top.indices.tolist()))
        prof = "  per-layer mean weight (%): " + " ".join(f"{v*100:4.1f}" for v in w_mean.tolist())
        print(prof)

    print(f"\ncorrection magnitude ||z - equalmean|| / ||equalmean|| (mean over tokens): "
          f"{corr_ratio_sum / n_tokens:.4f}")
    print("  (how far the learned depth-attention moves the latent off the plain 1/K mean floor;")
    print("   ~0 => attention is a near-no-op and the latent IS the uniform mean, collapse-proof by construction)")

    if not args.no_loo:
        loo = (loo_shift_sum / n_imgs).cpu()                  # [K] relative latent shift
        order = torch.argsort(loo, descending=True)
        print(f"\nLEAVE-ONE-OUT relative latent shift ||z_full - z_drop_k|| / ||z_full|| per layer:")
        print("  " + " ".join(f"{v:5.3f}" for v in loo.tolist()))
        print("  most-relied layers (largest shift when dropped): " + ", ".join(
            f"L{FUSION_PARAMS['layers'][j]}={loo[j]:.3f}" for j in order[:6].tolist()))
        shallow_frac = float(loo[0] / loo.sum())
        print(f"  shallowest-layer (L1) share of total LOO reliance: {shallow_frac*100:.1f}%  "
              f"(uniform would be {100.0/K:.1f}%)")

    print("\nVERDICT GUIDE: collapse == block norm-entropy ~0 AND idx0/L1 weight -> ~100% AND")
    print("L1 dominating the LOO reliance. Healthy == entropy well above 0, weight spread across")
    print("layers (esp. some mass on mid/deep), LOO reliance not dominated by L1.")


if __name__ == "__main__":
    main()
