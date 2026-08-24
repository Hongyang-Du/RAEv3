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
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})
import numpy as np
import torch
from omegaconf import OmegaConf

from utils.model_utils import get_obj_from_str, instantiate_from_config
from encoders.vision_encoder import create_encoder
from stage1.rae import _load_decoder

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-images", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
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
    enc.to(torch.bfloat16)        # bf16 weights to fit alongside a running job (compute is bf16 anyway)

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
        combine.to(torch.bfloat16); dec.to(torch.bfloat16)   # bf16 weights to fit beside a running job
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

    def load_official(cfg_path="configs/stage1/sampling/dinov3l-k23-imagenet.yaml"):
        # Official raev2 dinov3l-k23 decoder (fixed mean + L23 token-mean surrogate,
        # stats-normalized latent). Same subset-probe interface recon(toks, S).
        cfg = OmegaConf.load(cfg_path)
        rae = instantiate_from_config(cfg.stage_1).to(device).eval()
        for p in rae.parameters():
            p.requires_grad_(False)
        rae.to(torch.bfloat16)

        def recon(toks, S):
            if len(S) == 0:                                   # empty set -> zero latent baseline
                z = torch.zeros(toks[0].shape[0], toks[0].shape[1], 1024,
                                device=device, dtype=toks[0].dtype)
            else:
                z = torch.stack([toks[i] for i in S]).mean(0) \
                    + toks[-1].mean(dim=1, keepdim=True)       # FIXED L23 surrogate
            b, n, c = z.shape
            z = z.transpose(1, 2).view(b, c, 16, 16)
            if rae.do_normalization:
                z = (z - rae.latent_mean.to(device)) / torch.sqrt(rae.latent_var.to(device) + rae.eps)
            return rae.decode(z).clamp(0, 1)
        return recon

    models = {
        "RAEv2":          load_official(),
        "p0.9 (drop0.9)": load_ours("configs/stage1/decoder/infer-k23plain-p09.yaml"),
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
        std = [float(pc[k].std()) for k in sizes]                       # std over images x subsets
        p10 = [float(np.percentile(pc[k], 10)) for k in sizes]
        p90 = [float(np.percentile(pc[k], 90)) for k in sizes]
        marg_db = [curve[i] - curve[i - 1] for i in range(1, len(sizes))]
        mse_mean = [float(mc[k].mean()) for k in sizes]
        marg_mse = [mse_mean[i - 1] - mse_mean[i] for i in range(1, len(sizes))]   # MSE reduction
        shap = (shap_sum[m] / np.maximum(shap_cnt[m], 1)).tolist()
        res[m] = {"curve": curve, "std": std, "p10": p10, "p90": p90, "marg_db": marg_db,
                  "marg_mse": marg_mse, "shapley": shap}
        print(f"[{m}] v(1)={curve[0]:.2f}  v(N)={curve[-1]:.2f} dB   std@k=2={std[1]:.2f}", flush=True)

    # ---- plot: 4 SEPARATE figures ----
    colors = {"RAEv2": "gray", "p0.9 (drop0.9)": "#F6850C"}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    base = args.out[:-4] if args.out.endswith(".png") else args.out

    def save(fig, suffix):
        for ext in ("png", "pdf"):
            out = f"{base}_{suffix}.{ext}"
            fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"saved -> {base}_{suffix}.png")
        plt.close(fig)

    # 1) value v(S) vs |S|  (line = mean, band = mean +/- std over images x subsets)
    fig, a = plt.subplots(figsize=(7.5, 5))
    for m in models:
        col = colors[m]
        cu = np.array(res[m]["curve"]); sd = np.array(res[m]["std"])
        a.plot(sizes, cu, "o-", ms=3, color=col, label=m)
        a.fill_between(sizes, cu - sd, cu + sd, color=col, alpha=0.15)
    a.set_title("Value  v(S) = PSNR  vs  |S|   (line = mean, band = ± std)")
    a.set_xlabel("|S| (number of layers)"); a.set_ylabel("PSNR [dB]")
    a.grid(alpha=0.3); a.legend()
    save(fig, "1_value")

    # 2) marginal d(k) in dB
    fig, b = plt.subplots(figsize=(7.5, 5))
    for m in models:
        b.plot(sizes[1:], res[m]["marg_db"], "o-", ms=3, color=colors[m], label=m)
    b.set_title("Marginal  d(k) = v(k) - v(k-1)  [dB]   (decreasing = submodular/redundant)")
    b.set_xlabel("k-th layer added"); b.set_ylabel("ΔPSNR [dB]")
    b.grid(alpha=0.3); b.legend()
    save(fig, "2_marginal_db")

    # 3) marginal MSE reduction
    fig, c = plt.subplots(figsize=(7.5, 5))
    for m in models:
        c.plot(sizes[1:], res[m]["marg_mse"], "o-", ms=3, color=colors[m], label=m)
    c.set_title("Marginal  MSE reduction  [MSE domain]   (scale-dependence check)")
    c.set_xlabel("k-th layer added"); c.set_ylabel("MSE(k-1) - MSE(k)")
    c.grid(alpha=0.3); c.legend()
    save(fig, "3_marginal_mse")

    # 4) Monte-Carlo Shapley per layer
    fig, d = plt.subplots(figsize=(7.5, 5))
    for m in models:
        d.plot(LAYERS, res[m]["shapley"], "o-", ms=3, color=colors[m], label=m)
    d.set_title("Monte-Carlo Shapley per layer  (avg marginal contribution, dB)")
    d.set_xlabel("DINOv3 layer"); d.set_ylabel("Shapley [dB]")
    d.grid(alpha=0.3); d.legend()
    save(fig, "4_shapley")

    with open(base + ".json", "w") as f:
        json.dump({"layers": LAYERS, "num_images": args.num_images, "perms": args.perms,
                   "results": res}, f, indent=2)


if __name__ == "__main__":
    main()
