#!/usr/bin/env python3
"""Stage-0 exit gate: is the JEPA-trained fusion latent healthy BEFORE we spend a
decoder on it? Loads a train_fusion_jepa.py checkpoint (combine weights) and reports,
on N real val images with NO decoder:

  1. latent complexity of z_full (the eval/full-feed latent) -- the exact metrics
     that diagnosed the k23 failure: effective dim (PR), spatial smoothness (adjacent
     16x16-token cos), mean |excess kurtosis|. Compared against the param-free
     sum-mean(+cls) baseline from the SAME encoder pass. Target: smoothness/kurt at or
     better than the sum-mean baseline (the recon-trained depth-attn was ~2x rougher /
     ~3.7x heavier-tailed).
  2. pred-loss bucketed by deepest-dropped layer (needs the predictor from the ckpt):
     dropping a DEEP layer should cost MORE than a shallow one. Flat/inverted => z_full
     doesn't encode deep-only info.
  3. per-layer attention weights (head+token+image averaged) via the combine's
     return_attn probe path: JEPA should NOT collapse onto the shallowest layer the
     way the recon objective did.

Linear probe (target >=85) is left to the existing run_linear_probes.sh against the
combine in the ckpt; this script covers only the decoder-free signatures.

Usage:
    python src/probe_fusion_jepa.py --ckpt ckpt/stage0-jepa-k23/ckpt_latest.pt \
        --data-dir /mnt/localssd/imagenet-256 [--n 1024 --bs 16 --p-drop 0.3]
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from encoders.vision_encoder import create_encoder
from stage1.mask_cond import sample_stratified_masks
from stage1.jepa_predictor import JepaPredictor
from utils.model_utils import get_obj_from_str


def complexity(name, Z, gh=16):
    """Z: [Nimg, N, C] torch. Same math as gfid_eval/latent_complexity.py."""
    Z = Z.numpy().astype(np.float64)
    Nimg = Z.shape[0]
    X = Z.reshape(-1, Z.shape[-1])
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    Cm = (Xs.T @ Xs) / Xs.shape[0]
    ev = np.clip(np.linalg.eigvalsh(Cm), 0, None)[::-1]
    pr = (ev.sum() ** 2) / (ev ** 2).sum()
    kurt = (Xs ** 4).mean(0) - 3.0
    G = (Z.reshape(Nimg, gh, gh, Z.shape[-1]) - mu) / sd

    def cos(a, b):
        num = (a * b).sum(-1)
        den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
        return (num / den).mean()
    sm = 0.5 * (cos(G[:, :, :-1], G[:, :, 1:]) + cos(G[:, :-1, :], G[:, 1:, :]))
    out = dict(pr=float(pr), smooth=float(sm), kurt_abs=float(np.abs(kurt).mean()))
    print(f"[{name}]  eff-dim(PR)={out['pr']:.2f}  spatial-cos={out['smooth']:.4f}  "
          f"|kurt|={out['kurt_abs']:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True, help="dir with imagenet-latents-images arrow OR train/")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--p-drop", type=float, default=0.3)
    args = ap.parse_args()
    dev = torch.device("cuda:0")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    C = cfg["combine"]
    layers = ck["layers"]
    K = len(layers)
    dim = int(C["params"].get("dim", 1024))
    print(f"ckpt ep{ck.get('epoch')} step{ck.get('global_step')}  K={K} dim={dim} layers={layers}")

    combine = get_obj_from_str(C["target"])(**C["params"]).to(dev)
    combine.load_state_dict(ck["combine"]); combine.eval()              # no-EMA: raw combine weights
    predictor = None
    if "predictor" in ck:
        P = cfg["predictor"]
        nt = ck["predictor"]["pos_emb"].shape[1] if "pos_emb" in ck["predictor"] else 0
        predictor = JepaPredictor(dim, K, P["n_layers"], P["n_heads"], P["mlp_mult"],
                                  num_tokens=nt).to(dev)
        predictor.load_state_dict(ck["predictor"]); predictor.eval()

    enc_mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    enc_std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    layers_str = ".".join(str(x) for x in layers)
    encoder = create_encoder(f"dinov3mls-vit-l16[layers={layers_str}]",
                             device=dev, resolution=args.image_size)
    encoder.eval()

    @torch.no_grad()
    def encode_layers(x):
        x = (x - enc_mean) / enc_std
        return list(encoder.model.get_intermediate_layers(
            x, n=encoder.layer_indices, reshape=False, return_class_token=False, norm=True))

    # data: reuse the same loaders the trainer uses
    tf = transforms.Compose([transforms.Resize(args.image_size + 32),
                             transforms.CenterCrop(args.image_size), transforms.ToTensor()])
    arrow = os.path.join(args.data_dir, "imagenet-latents-images")
    if os.path.isdir(arrow):
        from datasets import load_from_disk
        ds = load_from_disk(arrow)
        if hasattr(ds, "keys"):
            ds = ds[list(ds.keys())[0]]
        col = "image" if "image" in ds.column_names else ds.column_names[0]
        def get(i): return tf(ds[i][col].convert("RGB"))
    else:
        from torchvision.datasets import ImageFolder
        ds = ImageFolder(os.path.join(args.data_dir, "train"), transform=tf)
        def get(i): return ds[i][0]

    Zfull, Zsum = [], []
    bucket_sum = torch.zeros(K, device=dev); bucket_cnt = torch.zeros(K, device=dev)
    attn_acc = torch.zeros(K, device=dev); attn_n = 0
    n = 0
    while n < args.n:
        imgs = torch.stack([get(i) for i in range(n, min(n + args.bs, args.n))]).to(dev)
        B = imgs.shape[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lt = encode_layers(imgs)
            stk = torch.stack(lt, 0).float()                            # [K,B,N,C]
            # baseline: param-free sum-mean (+ cls surrogate if the combine uses it)
            z_sum = stk.mean(0)
            if getattr(combine, "cls_surrogate", False):
                z_sum = z_sum + stk[-1].mean(1, keepdim=True)
            z_full, attns = combine(lt, return_attn=True)               # full feed + attn weights
        Zfull.append(z_full.float().cpu()); Zsum.append(z_sum.cpu())
        # full-feed attn: average |weights| over blocks, tokens, images -> [K]
        attn_acc += torch.stack(attns, 0).float().mean(0).mean(1).sum(0)  # sum over B of mean over N
        attn_n += B

        # bucketed pred-loss under random subsets (predictor needed)
        if predictor is not None:
            mask_ctx = sample_stratified_masks(B, K, args.p_drop, full_frac=0.0,
                                               uniform_frac=0.0, device=dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                z_ctx = combine(lt, mask=mask_ctx)
                z_hat = predictor(z_ctx, mask_ctx)
            err = F.smooth_l1_loss(z_hat.float(), z_full.float(), reduction="none").mean(dim=(1, 2))
            dropped = ~mask_ctx.bool()
            idxK = torch.arange(K, device=dev).view(1, K)
            deepest = torch.where(dropped, idxK, torch.full_like(idxK, -1)).max(1).values
            v = deepest >= 0
            if v.any():
                bucket_sum.index_add_(0, deepest[v], err[v])
                bucket_cnt.index_add_(0, deepest[v], torch.ones(int(v.sum()), device=dev))
        n += B
        if n % 256 == 0:
            print(f"  encoded {n}/{args.n}")

    print(f"\n=== latent complexity (N={n}) ===")
    d = complexity("depth-attn JEPA z_full", torch.cat(Zfull, 0))
    r = complexity("param-free sum-mean", torch.cat(Zsum, 0))
    print("  ratio JEPA/sum-mean:  " + "  ".join(
        f"{k}={d[k]/r[k]:.3f}" for k in ("pr", "smooth", "kurt_abs")))

    aw = (attn_acc / max(1, attn_n)).tolist()
    print(f"\n=== full-feed per-layer attention weight (avg heads/tokens/imgs) ===")
    print("  " + "  ".join(f"L{l}:{w:.3f}" for l, w in zip(layers, aw)))
    print(f"  argmax layer = L{layers[int(np.argmax(aw))]}  (shallow-collapse would peak at L{layers[0]})")

    if predictor is not None:
        means = (bucket_sum / bucket_cnt.clamp_min(1)).tolist()
        print(f"\n=== pred-loss by deepest-dropped layer (p_drop={args.p_drop}) ===")
        print("  " + "  ".join(f"L{l}:{m:.3e}" for l, m in zip(layers, means)))
        print("  (want INCREASING toward deep layers: deep info is harder to recover)")


if __name__ == "__main__":
    main()
