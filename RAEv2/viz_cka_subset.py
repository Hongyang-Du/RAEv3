"""Encoder x Decoder linear-CKA under different input-layer subsets.

Grid: rows = decoder (OmniRAE / RAEv2), cols = input subset fed to the decoder
(k23 / k7 / l11). Each cell = CKA[encoder layer 1..23  x  decoder block 0..28].
Encoder activations are shared (DINOv3 always produces all 23 layers); only the
decoder input changes. Shows whether the encoder<->decoder alignment is stable
when fewer layers are fed (robustness) or degrades.

  python viz_cka_subset.py --num 256 --batch 64 --out assets/viz_pca/cka_subset.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LAYERS = list(range(1, 24))
NB = 29
CMAP = "viridis"
SUBSETS = [("k23", None, "input k=23 (all)"),
           ("k7", [10, 12, 14, 16, 18, 20, 22], "input k=7 (11,..,23)"),
           ("l11", [10], "input l=11 (single)")]
MODELS = {
    "ours": dict(ckpt="output_full/decoder_random_drop_layer_mls_plain_cls_k23_16ep/ckpt_ep016.pt",
                 weighting="random_drop", cls_surrogate=True, title="OmniRAE (ours)"),
    "raev2": dict(ckpt="output_full/decoder_raev2_k23/ckpt_latest.pt",
                  weighting="mean", cls_surrogate=False, title="RAEv2 K=23"),
}

from stage1.rae import _load_decoder                       # noqa: E402
from stage1.combine import MLSCombine                      # noqa: E402
from encoders.vision_encoder import create_encoder         # noqa: E402


def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y) ** 2
    return float(hsic / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="assets/viz_pca/cka_subset.png")
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

    enc_acts = [[] for _ in LAYERS]
    dec_acts = {(m, s): [[] for _ in range(NB)] for m in MODELS for s, _i, _t in SUBSETS}
    n_done = 0
    for i in range(0, len(sel), args.batch):
        bi = sel[i:i + args.batch]
        imgs = torch.stack([torch.from_numpy(arr[j].copy()) for j in bi])
        imgs = imgs.permute(0, 3, 1, 2).float().to(dev) / 255
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            toks = list(enc.model.get_intermediate_layers(
                (imgs - mean) / std, n=LAYERS, reshape=False,
                return_class_token=False, norm=True))
            for li in range(len(LAYERS)):
                enc_acts[li].append(toks[li].mean(1).float().cpu())
            for m, (cb, dec) in decs.items():
                for s, idx, _t in SUBSETS:
                    out = dec(cb(toks, idx=idx), drop_cls_token=False, output_hidden_states=True)
                    for j in range(NB):
                        dec_acts[(m, s)][j].append(out.hidden_states[j][:, 1:, :].mean(1).float().cpu())
        n_done += len(bi)
        print(f"  {n_done}/{len(sel)}", flush=True)

    E = [torch.cat(a, 0).numpy() for a in enc_acts]
    cka = {}
    for m in MODELS:
        for s, _i, _t in SUBSETS:
            D = [torch.cat(a, 0).numpy() for a in dec_acts[(m, s)]]
            cka[(m, s)] = np.array([[linear_cka(E[li], D[j]) for j in range(NB)]
                                    for li in range(len(LAYERS))])

    vmax = max(v.max() for v in cka.values())
    vmin = min(v.min() for v in cka.values())
    mlist = list(MODELS)
    fig = plt.figure(figsize=(4.7 * len(SUBSETS) + 0.8, 4.4 * len(mlist)))
    gs = GridSpec(len(mlist), len(SUBSETS) + 1, figure=fig,
                  width_ratios=[1] * len(SUBSETS) + [0.045], wspace=0.10, hspace=0.18)
    for ri, m in enumerate(mlist):
        for ci, (s, _i, stitle) in enumerate(SUBSETS):
            ax = fig.add_subplot(gs[ri, ci])
            ax.imshow(cka[(m, s)], cmap=CMAP, vmin=vmin, vmax=vmax, aspect="auto",
                      extent=[-0.5, NB - 0.5, len(LAYERS) + 0.5, 0.5])
            if ri == 0:
                ax.set_title(stitle, fontsize=12)
            if ri == len(mlist) - 1:
                ax.set_xlabel("decoder block (0=embed .. 28)")
            if ci == 0:
                ax.set_ylabel(f"{MODELS[m]['title']}\nDINOv3 enc layer 1..23", fontsize=11)
            ax.set_xticks(range(0, NB, 6)); ax.set_yticks(range(1, len(LAYERS) + 1, 4))
    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(ScalarMappable(cmap=CMAP), cax=cax); cb.mappable.set_clim(vmin, vmax)
    cb.set_label("linear CKA", fontsize=12, rotation=90)
    fig.suptitle(f"Encoder x Decoder CKA under input-layer subsets (N={n_done})", fontsize=14, y=0.995)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    fig.savefig(args.out.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
