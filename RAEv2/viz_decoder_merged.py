"""Merged robustness figure for one image + one input-layer subset:

    [ original ] | [ 4 sampled decoder-block PCA->RGB, ours vs raev2 ] | [ relL2 error curve ]

Feed ONLY the chosen subset (e.g. L11 or k7) to both decoders, measure each
decoder block's feature error vs the native(23) trajectory, and PCA 4 sampled
blocks to RGB (basis fit on native, so color == distance from native). The curve
(ours=orange, raev2=gray) shows whether our late blocks "heal" the perturbation.

  python viz_decoder_merged.py --img assets/samples/cat.jpg --subset L11 --out ...
  python viz_decoder_merged.py --img npz:14601 --subset k7 --label tiger --out ...

CPU, single image.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from encoders.vision_encoder import create_encoder
from stage1.rae import _load_decoder
from stage1.combine import MLSCombine

DEVICE = "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LAYERS = list(range(1, 24))
GRID = 16
ALL_BLOCKS = list(range(0, 29))
PCA_BLOCKS = [7, 14, 21, 28]                      # 4 sampled blocks (early / peak / recover / final)

SUBSET_IDX = {
    "k7": [10, 12, 14, 16, 18, 20, 22],          # layers 11,13,..,23
    "L11": [10],                                 # layer 11 only
}

MODELS = {
    "ours": dict(
        ckpt="output_full/decoder_random_drop_layer_mls_plain_cls_k23_16ep_gan8/ckpt_latest.pt",
        weighting="random_drop", cls_surrogate=True, color="#ff7f0e"),    # orange
    "raev2": dict(
        ckpt="output_full/decoder_raev2_k23/ckpt_latest.pt",
        weighting="mean", cls_surrogate=False, color="0.45"),             # gray
}
MODEL_TITLE = {"ours": "ours (random-drop+cls)", "raev2": "RAEv2 k23 (fixed mean)"}


def load_image(spec):
    if spec.startswith("npz:"):
        idx = int(spec.split(":", 1)[1])
        arr = np.load("data_eval/imagenet-256-val.npz", mmap_mode="r")["arr_0"]
        img = Image.fromarray(arr[idx].copy())
    else:
        img = Image.open(spec).convert("RGB").resize((256, 256), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0), np.asarray(img)


@torch.no_grad()
def encode_tokens(enc, x):
    return list(enc.model.get_intermediate_layers(
        (x - MEAN) / STD, n=LAYERS, reshape=False,
        return_class_token=False, norm=True))


@torch.no_grad()
def decoder_feats(dec, combine, toks, idx):
    z = combine(toks, idx=idx)
    out = dec(z, drop_cls_token=False, output_hidden_states=True)
    rec = (dec.unpatchify(out.logits) * STD + MEAN).clamp(0, 1)
    rec = rec[0].permute(1, 2, 0).mul(255).round().byte().numpy()
    feats = {b: out.hidden_states[b][0, 1:, :].float() for b in ALL_BLOCKS}
    return rec, feats


def fit_pca_rgb(native_feat):
    mu = native_feat.mean(0)
    Xc = native_feat - mu
    _, _, V = torch.linalg.svd(Xc, full_matrices=False)
    comps = V[:3]
    pn = Xc @ comps.T
    lo = torch.quantile(pn, 0.01, dim=0)
    hi = torch.quantile(pn, 0.99, dim=0)
    rng = (hi - lo).clamp_min(1e-6)

    def project(feat):
        p = (((feat - mu) @ comps.T - lo) / rng).clamp(0, 1)
        return p.reshape(GRID, GRID, 3).numpy()
    return project


def to_disp(rgb):
    return np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="file path or npz:IDX")
    ap.add_argument("--subset", required=True, choices=list(SUBSET_IDX))
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    idx = SUBSET_IDX[args.subset]
    label = args.label or os.path.splitext(os.path.basename(args.img))[0]

    x, img_np = load_image(args.img)
    enc = create_encoder("dinov3mls-vit-l16[layers=" + ".".join(map(str, LAYERS)) + "]",
                         device=DEVICE, resolution=256).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    toks = encode_tokens(enc, x)

    res = {}
    for m, mc in MODELS.items():
        ck = torch.load(mc["ckpt"], map_location="cpu", weights_only=False)
        combine = MLSCombine(layers=LAYERS, weighting=mc["weighting"],
                             cls_surrogate=mc["cls_surrogate"], projector="none",
                             dim=1024, out_dim=1024).to(DEVICE).eval()
        combine.load_state_dict(ck["ema_combine"])
        dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                            num_patches=256, pretrained_path=None).to(DEVICE).eval()
        dec.load_state_dict(ck["ema_dec"])
        del ck
        _, nat = decoder_feats(dec, combine, toks, None)
        rec, sub = decoder_feats(dec, combine, toks, idx)
        proj = {b: fit_pca_rgb(nat[b]) for b in PCA_BLOCKS}
        rell2 = {b: ((sub[b] - nat[b]).norm(dim=-1) /
                     nat[b].norm(dim=-1).clamp_min(1e-6)).mean().item() for b in ALL_BLOCKS}
        res[m] = dict(rec=rec, sub=sub, proj=proj, rell2=rell2)
        print(f"[{m}] done", flush=True)

    # ---- layout: original | 2x4 PCA | curve ----
    fig = plt.figure(figsize=(17, 5.2))
    gs = GridSpec(2, 7, figure=fig, width_ratios=[1.25, 1, 1, 1, 1, 0.15, 2.6],
                  hspace=0.18, wspace=0.08)

    # original (spans both rows, col 0)
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(img_np); ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_title(f"input: {label}\n(feed-{args.subset})", fontsize=11)

    # PCA grid: rows = models, cols 1..4 = PCA_BLOCKS
    for ri, m in enumerate(["ours", "raev2"]):
        for ci, b in enumerate(PCA_BLOCKS):
            ax = fig.add_subplot(gs[ri, 1 + ci])
            ax.imshow(to_disp(res[m]["proj"][b](res[m]["sub"][b])))
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"block {b}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(MODEL_TITLE[m], fontsize=10,
                              color=MODELS[m]["color"] if m == "ours" else "0.3")
            ax.text(0.5, -0.02, f"relL2={res[m]['rell2'][b]:.2f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=8,
                    color=MODELS[m]["color"] if m == "ours" else "0.3")

    # curve (spans both rows, col 6)
    axc = fig.add_subplot(gs[:, 6])
    for m in ["ours", "raev2"]:
        ys = [res[m]["rell2"][b] for b in ALL_BLOCKS]
        axc.plot(ALL_BLOCKS, ys, marker="o", ms=3, lw=2.0,
                 color=MODELS[m]["color"], label=MODEL_TITLE[m])
    for b in PCA_BLOCKS:
        axc.axvline(b, color="0.8", ls="--", lw=0.8, zorder=0)
    axc.set_xlabel("decoder block (0=embed .. 28)")
    axc.set_ylabel("feature error vs native  (relative L2, lower=robust)")
    axc.set_title(f"per-block error, feed-{args.subset}", fontsize=11)
    axc.grid(alpha=0.3); axc.legend(fontsize=9, loc="upper right")
    axc.set_ylim(bottom=0)

    fig.suptitle(f"Decoder robustness to feeding only a layer subset — {label}, feed-{args.subset}",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
