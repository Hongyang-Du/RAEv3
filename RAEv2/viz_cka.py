"""Encoder x Decoder linear-CKA heatmaps (ours vs RAEv2).

For N val images: take DINOv3 encoder layer activations (1..23) and decoder block
activations (0..28, native k23 latent), image-level mean-pooled to [N,d]; compute
linear CKA between every (encoder layer, decoder block) pair. Encoder is shared,
so we draw two side-by-side maps: encoder x OmniRAE-decoder and encoder x RAEv2.

  python viz_cka.py --num 256 --batch 64 --out assets/viz_pca/cka_enc_dec.png
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
LAYERS = list(range(1, 24))                          # encoder layers
NB = 29                                              # decoder hidden_states (0=embed..28)
CMAP = "viridis"
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
    """X [N,d1], Y [N,d2] -> linear CKA in [0,1] (column-centered)."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y) ** 2
    nx = np.linalg.norm(X.T @ X)
    ny = np.linalg.norm(Y.T @ Y)
    return float(hsic / (nx * ny + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="assets/viz_pca/cka_enc_dec.png")
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

    enc_acts = [[] for _ in LAYERS]                  # per enc layer: list of [B,1024]
    dec_acts = {m: [[] for _ in range(NB)] for m in MODELS}
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
                enc_acts[li].append(toks[li].mean(1).float().cpu())     # [B,1024]
            for m, (cb, dec) in decs.items():
                out = dec(cb(toks, idx=None), drop_cls_token=False, output_hidden_states=True)
                for j in range(NB):
                    dec_acts[m][j].append(out.hidden_states[j][:, 1:, :].mean(1).float().cpu())
        n_done += len(bi)
        print(f"  {n_done}/{len(sel)}", flush=True)

    E = [torch.cat(a, 0).numpy() for a in enc_acts]                 # 23 x [N,1024]
    D = {m: [torch.cat(a, 0).numpy() for a in dec_acts[m]] for m in MODELS}

    cka = {m: np.array([[linear_cka(E[li], D[m][j]) for j in range(NB)]
                        for li in range(len(LAYERS))]) for m in MODELS}

    # ---- two heatmaps side by side ----
    mlist = list(MODELS)
    fig = plt.figure(figsize=(6.6 * len(mlist) + 0.8, 6.0))
    gs = GridSpec(1, len(mlist) + 1, figure=fig, width_ratios=[1] * len(mlist) + [0.04], wspace=0.12)
    vmax = max(cka[m].max() for m in mlist)
    vmin = min(cka[m].min() for m in mlist)
    for ci, m in enumerate(mlist):
        ax = fig.add_subplot(gs[0, ci])
        ax.imshow(cka[m], cmap=CMAP, vmin=vmin, vmax=vmax, aspect="auto",
                  extent=[-0.5, NB - 0.5, len(LAYERS) + 0.5, 0.5])
        ax.set_title(MODELS[m]["title"], fontsize=13)
        ax.set_xlabel("decoder block (0=embed .. 28)")
        if ci == 0:
            ax.set_ylabel("DINOv3 encoder layer (1 .. 23)")
        ax.set_xticks(range(0, NB, 4)); ax.set_yticks(range(1, len(LAYERS) + 1, 2))
    cax = fig.add_subplot(gs[0, -1])
    cb = fig.colorbar(ScalarMappable(cmap=CMAP), cax=cax)
    cb.set_label("linear CKA", fontsize=12, rotation=90)
    cb.mappable.set_clim(vmin, vmax)
    fig.suptitle(f"Encoder x Decoder representational similarity (linear CKA, N={n_done})",
                 fontsize=13, y=0.99)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    fig.savefig(args.out.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
