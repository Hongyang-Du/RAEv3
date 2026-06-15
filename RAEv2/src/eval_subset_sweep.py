#!/usr/bin/env python3
"""Coalition value-function sweep over layer-subset SIZE for our k=23 decoders.

Value function  v(S) = PSNR( D( mean_{i in S} LN(h_i(x)) ) )  -- exactly the latent
the decoder consumes (post-LN mean, no extra normalization). We sweep |S|=k=1..N by
PERMUTATION sampling: P random permutations, evaluate every prefix. One sampling
yields all three diagnostics, on the SAME subsets for every model:

  1. value curve     v vs |S|   (+ 10/90 pct band: band wide = composition matters
                                  = specialized; narrow = layers interchangeable)
  2. marginal curve  d(k)=v(k)-v(k-1) vs k   (decreasing=submodular/redundant,
                                              increasing=supermodular/complementary)
  3. Monte-Carlo Shapley per layer (avg marginal of adding that layer)

Submodularity label is scale-dependent, so the marginal is shown in BOTH PSNR(dB)
and MSE domains. Same val images + same permutations across all models.

Writes output_full/subset_sweep.png (+ .json).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from utils.model_utils import get_obj_from_str
from encoders.vision_encoder import create_encoder
from stage1.rae import _load_decoder

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-images", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--perms", type=int, default=64, help="random permutations (subset samples per size)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-npz", default="data_eval/imagenet-256-val.npz")
    ap.add_argument("--out", default="output_full/subset_sweep.png")
    args = ap.parse_args()
    device = torch.device("cuda")
    mean, std = MEAN.to(device), STD.to(device)

    LAYERS = list(range(1, 24))
    N = len(LAYERS)

    enc = create_encoder("dinov3mls-vit-l16[layers=" + ".".join(map(str, LAYERS)) + "]",
                         device=device, resolution=256).eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    def encode(imgs01):
        return list(enc.model.get_intermediate_layers(
            (imgs01 - mean) / std, n=LAYERS, reshape=False,
            return_class_token=False, norm=True))

    def load_ours(cfg_path):
        infer = OmegaConf.load(cfg_path)
        ck = torch.load(infer.eval.ckpt, map_location="cpu", weights_only=False)
        combine = get_obj_from_str(infer.combine.target)(
            **OmegaConf.to_container(infer.combine.params, resolve=True)).to(device).eval()
        combine.load_state_dict(ck["ema_combine"])
        dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                            num_patches=256, pretrained_path=None).to(device).eval()
        dec.load_state_dict(ck["ema_dec"])
        del ck

        def recon(toks, S):
            if len(S) == 0:                                   # empty set -> zero latent baseline
                z = torch.zeros(toks[0].shape[0], toks[0].shape[1], 1024,
                                device=device, dtype=toks[0].dtype)
            else:
                z = combine(toks, idx=list(S))
            out = dec(z, drop_cls_token=False).logits
            return (dec.unpatchify(out) * std + mean).clamp(0, 1)
        return recon

    models = {
        "ours_raev2":       load_ours("configs/stage1/decoder/infer-raev2k23.yaml"),
        "ours_drop+SIGReg": load_ours("configs/stage1/decoder/infer-k23-fed-k7.yaml"),
        "ours_drop_plain":  load_ours("configs/stage1/decoder/infer-k23plain-fed-k7.yaml"),
    }

    # fixed permutations (same subsets for every model)
    g = torch.Generator().manual_seed(args.seed)
    perms = [torch.randperm(N, generator=g).tolist() for _ in range(args.perms)]

    arr = np.load(args.val_npz, mmap_mode="r")
    arr = arr[arr.files[0]] if hasattr(arr, "files") else arr
    gi = torch.Generator().manual_seed(args.seed)
    idxs = torch.randperm(len(arr), generator=gi)[:args.num_images].tolist()

    # accumulators: per model -> psnr[k] list, mse[k] list, shapley sums per layer
    val_psnr = {m: {k: [] for k in range(1, N + 1)} for m in models}
    val_mse = {m: {k: [] for k in range(1, N + 1)} for m in models}
    shap_sum = {m: np.zeros(N) for m in models}
    shap_cnt = {m: np.zeros(N) for m in models}

    def metrics(rec, ref):
        mse = ((rec - ref) ** 2).flatten(1).mean(1)               # per image
        psnr = -10 * torch.log10(mse.clamp_min(1e-10))
        return psnr.float().cpu().numpy(), mse.float().cpu().numpy()

    n_done = 0
    with torch.no_grad():
        for i0 in range(0, len(idxs), args.batch):
            bi = idxs[i0:i0 + args.batch]
            imgs = torch.stack([torch.from_numpy(arr[j].copy()) for j in bi])
            imgs = imgs.permute(0, 3, 1, 2).float().to(device) / 255
            with torch.autocast("cuda", dtype=torch.bfloat16):
                toks = encode(imgs)
                for m, recon in models.items():
                    ps_empty, _ = metrics(recon(toks, []).float(), imgs)   # v(empty), once/batch
                    for perm in perms:
                        prev_psnr = ps_empty
                        for k in range(1, N + 1):
                            S = perm[:k]
                            ps, ms = metrics(recon(toks, S).float(), imgs)
                            val_psnr[m][k].append(ps)
                            val_mse[m][k].append(ms)
                            layer_added = perm[k - 1]
                            shap_sum[m][layer_added] += float((ps - prev_psnr).mean())
                            shap_cnt[m][layer_added] += 1
                            prev_psnr = ps
            n_done += imgs.shape[0]
            print(f"  {n_done}/{len(idxs)} images", flush=True)

    # aggregate
    res = {}
    sizes = list(range(1, N + 1))
    for m in models:
        pc = {k: np.concatenate(val_psnr[m][k]) for k in sizes}
        mc = {k: np.concatenate(val_mse[m][k]) for k in sizes}
        curve = [float(pc[k].mean()) for k in sizes]
        p10 = [float(np.percentile(pc[k], 10)) for k in sizes]
        p90 = [float(np.percentile(pc[k], 90)) for k in sizes]
        marg_db = [curve[i] - curve[i - 1] for i in range(1, len(sizes))]
        mse_mean = [float(mc[k].mean()) for k in sizes]
        marg_mse = [mse_mean[i - 1] - mse_mean[i] for i in range(1, len(sizes))]   # MSE reduction
        shap = (shap_sum[m] / np.maximum(shap_cnt[m], 1)).tolist()
        res[m] = {"curve": curve, "p10": p10, "p90": p90, "marg_db": marg_db,
                  "marg_mse": marg_mse, "shapley": shap}
        print(f"[{m}] v(1)={curve[0]:.2f}  v(N)={curve[-1]:.2f} dB   "
              f"band@k=2 = [{p10[1]:.1f},{p90[1]:.1f}]", flush=True)

    # ---- plot ----
    colors = {"ours_raev2": "tab:gray", "ours_drop+SIGReg": "tab:red", "ours_drop_plain": "tab:blue"}
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    a, b, c, d = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
    for m in models:
        col = colors[m]
        a.plot(sizes, res[m]["curve"], "o-", ms=3, color=col, label=m)
        a.fill_between(sizes, res[m]["p10"], res[m]["p90"], color=col, alpha=0.15)
        b.plot(sizes[1:], res[m]["marg_db"], "o-", ms=3, color=col, label=m)
        c.plot(sizes[1:], res[m]["marg_mse"], "o-", ms=3, color=col, label=m)
        d.plot(LAYERS, res[m]["shapley"], "o-", ms=3, color=col, label=m)
    a.set_title("Value  v(S) = PSNR  vs  |S|   (line=mean, band=10/90 pct over subsets)")
    a.set_xlabel("|S| (number of layers)"); a.set_ylabel("PSNR [dB]"); a.grid(alpha=0.3); a.legend(fontsize=8)
    b.set_title("Marginal  d(k) = v(k)-v(k-1)  [dB]   (decreasing = submodular/redundant)")
    b.set_xlabel("k-th layer added"); b.set_ylabel("dPSNR [dB]"); b.grid(alpha=0.3); b.legend(fontsize=8)
    c.set_title("Marginal  MSE reduction  [MSE domain]   (scale-dependence check)")
    c.set_xlabel("k-th layer added"); c.set_ylabel("MSE(k-1)-MSE(k)"); c.grid(alpha=0.3); c.legend(fontsize=8)
    d.set_title("Monte-Carlo Shapley per layer  (avg marginal contribution, dB)")
    d.set_xlabel("DINOv3 layer"); d.set_ylabel("Shapley [dB]"); d.grid(alpha=0.3); d.legend(fontsize=8)
    fig.suptitle(f"Layer-subset value sweep  (N={args.num_images} imgs x {args.perms} perms)", fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved -> {args.out}")
    with open(args.out.replace(".png", ".json"), "w") as f:
        json.dump({"layers": LAYERS, "num_images": args.num_images, "perms": args.perms,
                   "results": res}, f, indent=2)


if __name__ == "__main__":
    main()
