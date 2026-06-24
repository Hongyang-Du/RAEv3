"""Aggregate per-decoder-block feature error (relL2 vs native) over many val
images, for ours vs raev2, under feed-k7 and feed-L11. Saves a npz the figure
script reads. GPU, batched.

  python viz_stats_1000.py --num 1000 --batch 32 --out assets/viz_pca/stats_1000.npz
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch

from encoders.vision_encoder import create_encoder
from stage1.rae import _load_decoder
from stage1.combine import MLSCombine

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LAYERS = list(range(1, 24))
NB = 29                                            # decoder hidden_states (0=embed .. 28)
SUBSET_IDX = {"k7": [10, 12, 14, 16, 18, 20, 22], "L11": [10]}
MODELS = {
    "ours": dict(ckpt="output_full/decoder_random_drop_layer_mls_plain_cls_k23_16ep/ckpt_ep016.pt",
                 weighting="random_drop", cls_surrogate=True),
    "raev2": dict(ckpt="output_full/decoder_raev2_k23/ckpt_latest.pt",
                  weighting="mean", cls_surrogate=False),
}


@torch.no_grad()
def hidden(dec, z):
    out = dec(z, drop_cls_token=False, output_hidden_states=True)
    return [h[:, 1:, :].float() for h in out.hidden_states]   # drop cls, per block [B,256,D]


def rell2(sub, nat):
    return ((sub - nat).norm(dim=-1) / nat.norm(dim=-1).clamp_min(1e-6)).mean(dim=1)  # [B]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="assets/viz_pca/stats_1000.npz")
    args = ap.parse_args()
    dev = "cuda"

    arr = np.load("data_eval/imagenet-256-val.npz", mmap_mode="r")["arr_0"]
    g = torch.Generator().manual_seed(args.seed)
    sel = torch.randperm(len(arr), generator=g)[:args.num].tolist()

    enc = create_encoder("dinov3mls-vit-l16[layers=" + ".".join(map(str, LAYERS)) + "]",
                         device=dev, resolution=256).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    mean, std = MEAN.to(dev), STD.to(dev)

    decs = {}
    for m, mc in MODELS.items():
        ck = torch.load(mc["ckpt"], map_location="cpu", weights_only=False)
        cb = MLSCombine(layers=LAYERS, weighting=mc["weighting"], cls_surrogate=mc["cls_surrogate"],
                        projector="none", dim=1024, out_dim=1024).to(dev).eval()
        cb.load_state_dict(ck["ema_combine"])
        dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                            num_patches=256, pretrained_path=None).to(dev).eval()
        dec.load_state_dict(ck["ema_dec"])
        decs[m] = (cb, dec)
        del ck

    # accumulators: [model][subset] -> sum, sumsq over images, per block
    ssum = {m: {s: np.zeros(NB) for s in SUBSET_IDX} for m in MODELS}
    ssq = {m: {s: np.zeros(NB) for s in SUBSET_IDX} for m in MODELS}
    n_done = 0

    for i in range(0, len(sel), args.batch):
        bi = sel[i:i + args.batch]
        imgs = torch.stack([torch.from_numpy(arr[j].copy()) for j in bi])
        imgs = imgs.permute(0, 3, 1, 2).float().to(dev) / 255
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            toks = list(enc.model.get_intermediate_layers(
                (imgs - mean) / std, n=LAYERS, reshape=False,
                return_class_token=False, norm=True))
            for m, (cb, dec) in decs.items():
                nat = hidden(dec, cb(toks, idx=None))
                for s, idx in SUBSET_IDX.items():
                    sub = hidden(dec, cb(toks, idx=idx))
                    for b in range(NB):
                        v = rell2(sub[b], nat[b]).cpu().numpy()  # [B]
                        ssum[m][s][b] += v.sum()
                        ssq[m][s][b] += (v ** 2).sum()
        n_done += len(bi)
        print(f"  {n_done}/{len(sel)}", flush=True)

    mean_d, std_d = {}, {}
    for m in MODELS:
        for s in SUBSET_IDX:
            mu = ssum[m][s] / n_done
            var = (ssq[m][s] / n_done - mu ** 2).clip(min=0)
            mean_d[f"{m}__{s}__mean"] = mu
            std_d[f"{m}__{s}__std"] = np.sqrt(var)
    np.savez(args.out, n=n_done, blocks=np.arange(NB), **mean_d, **std_d)
    print(f"saved -> {args.out}  (N={n_done})", flush=True)
    for m in MODELS:
        for s in SUBSET_IDX:
            mu = mean_d[f"{m}__{s}__mean"]
            print(f"  {m:6s} {s:4s}  block28 relL2 = {mu[28]:.3f}  (peak {mu.max():.3f} @ {mu.argmax()})")


if __name__ == "__main__":
    main()
