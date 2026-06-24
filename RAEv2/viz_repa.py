"""REPA-Fig.6-style feature-affinity figures (multi-scene).

For each image, columns = Target (frozen DINOv3 enc.) | ours (feed-SUBSET) |
RAEv2 (feed-SUBSET); affinity = cosine sim of a *-marked query patch to all
patches, smoothly overlaid, shared blue-green colorbar. Uses a DEEP decoder
block (where the fixed-mean RAEv2 features have diverged).

Saves, for the given subset+block:
  * per image: repa_<label>_<subset>_b<block>.png   (rows = subjectA/subjectB/background)
  * multi-scene: repa_multiscene_<subset>_b<block>.png  (rows = images, one subject query each)

  python viz_repa.py --subset L11 --block 24
  python viz_repa.py --subset k7  --block 24
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

DEVICE = "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LAYERS = list(range(1, 24))
GRID = 16
SUBSET_IDX = {"k7": [10, 12, 14, 16, 18, 20, 22], "L11": [10]}
STAR_COLORS = ["#ff2b2b", "#ffd400", "#ffffff"]    # subjectA, subjectB, background
ROW_NAMES = ["subject A", "subject B", "background"]
CMAP = LinearSegmentedColormap.from_list(
    "bluegreen", ["#06101c", "#0b3b54", "#10746e", "#3fbf8f", "#d6f5cf"])
OUT_DIR = "assets/viz_pca"
IMAGES = [("npz:14200", "cat"), ("npz:13150", "corgi"), ("npz:851", "jay")]
MODELS = {
    "ours": dict(ckpt="output_full/decoder_random_drop_layer_mls_plain_cls_k23_16ep/ckpt_ep016.pt",
                 weighting="random_drop", cls_surrogate=True),
    "raev2": dict(ckpt="output_full/decoder_raev2_k23/ckpt_latest.pt",
                  weighting="mean", cls_surrogate=False),
}

from stage1.rae import _load_decoder                       # noqa: E402
from stage1.combine import MLSCombine                      # noqa: E402
from encoders.vision_encoder import create_encoder         # noqa: E402


def load_image(spec):
    if spec.startswith("npz:"):
        idx = int(spec.split(":", 1)[1])
        arr = np.load("data_eval/imagenet-256-val.npz", mmap_mode="r")["arr_0"]
        img = Image.fromarray(arr[idx].copy())
    else:
        img = Image.open(spec).convert("RGB").resize((256, 256), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0), np.asarray(img)


def affinity(feat, q):
    """cosine sim of query q to all patches -> 256x256 bicubic, [2,98]-pct stretch."""
    fn = F.normalize(feat, dim=-1)
    aff = (fn @ fn[q]).reshape(1, 1, GRID, GRID)
    aff = F.interpolate(aff, size=(256, 256), mode="bicubic", align_corners=False)[0, 0].numpy()
    lo, hi = np.percentile(aff, 2), np.percentile(aff, 98)
    return np.clip((aff - lo) / max(hi - lo, 1e-6), 0, 1)


def pick_queries(target_feat):
    norm = target_feat.norm(dim=-1).reshape(GRID, GRID).numpy()
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    gw = np.exp(-(((yy - 7.5) ** 2 + (xx - 7.5) ** 2) / (2 * 4.5 ** 2)))
    sal = norm * gw
    qa = int(sal.argmax()); ra, ca = qa // GRID, qa % GRID
    s2 = sal.copy(); s2[(np.abs(yy - ra) + np.abs(xx - ca)) < 4] = -np.inf
    qb = int(s2.argmax()); rb, cb = qb // GRID, qb % GRID
    fb = ((np.abs(yy - ra) + np.abs(xx - ca)) >= 5) & ((np.abs(yy - rb) + np.abs(xx - cb)) >= 5)
    sc = norm.copy(); sc[~fb] = np.inf
    qg = int(sc.argmin()); rg, cg = qg // GRID, qg % GRID
    return [(qa, ra, ca), (qb, rb, cb), (qg, rg, cg)]


def build(enc, decs, spec, subset_idx, block):
    x, img_np = load_image(spec)
    toks = list(enc.model.get_intermediate_layers(
        (x - MEAN) / STD, n=LAYERS, reshape=False, return_class_token=False, norm=True))
    feats = {"Target": toks[-1][0].float()}
    for m, (cb, dec) in decs.items():
        with torch.no_grad():
            out = dec(cb(toks, idx=subset_idx), drop_cls_token=False, output_hidden_states=True)
        feats[m] = out.hidden_states[block][0, 1:, :].float()
    return img_np, feats, pick_queries(feats["Target"])


def render(rows, col_titles, out):
    """rows: list of dict(img_np, feats, q, r, c, star, label, label_color)."""
    cols = ["Target", "ours", "raev2"]
    nrow, ncol = len(rows), len(cols)
    fig = plt.figure(figsize=(3.0 * ncol + 0.7, 3.0 * nrow))
    gs = GridSpec(nrow, ncol + 1, figure=fig,
                  width_ratios=[1] * ncol + [0.05], wspace=0.04, hspace=0.04)
    for ri, row in enumerate(rows):
        gray = np.asarray(Image.fromarray(row["img_np"]).convert("L").convert("RGB")) / 255.0
        for ci, key in enumerate(cols):
            ax = fig.add_subplot(gs[ri, ci])
            ax.imshow(0.30 * gray + 0.03)
            ax.imshow(affinity(row["feats"][key], row["q"]), cmap=CMAP, alpha=0.88, vmin=0, vmax=1)
            ax.scatter([(row["c"] + 0.5) * 16], [(row["r"] + 0.5) * 16], marker="*", s=300,
                       c=row["star"], edgecolors="black", linewidths=1.3, zorder=5)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(col_titles[ci], fontsize=12)
            if ci == 0:
                ax.set_ylabel(row["label"], fontsize=12, color=row["label_color"])
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(ScalarMappable(cmap=CMAP), cax=cax)
    cbar.set_label("feature affinity to query ★  (low → high)", fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="L11", choices=list(SUBSET_IDX))
    ap.add_argument("--block", type=int, default=24)
    args = ap.parse_args()
    idx = SUBSET_IDX[args.subset]
    os.makedirs(OUT_DIR, exist_ok=True)

    enc = create_encoder("dinov3mls-vit-l16[layers=" + ".".join(map(str, LAYERS)) + "]",
                         device=DEVICE, resolution=256).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    decs = {}
    for m, mc in MODELS.items():
        ck = torch.load(mc["ckpt"], map_location="cpu", weights_only=False)
        cb = MLSCombine(layers=LAYERS, weighting=mc["weighting"], cls_surrogate=mc["cls_surrogate"],
                        projector="none", dim=1024, out_dim=1024).to(DEVICE).eval()
        cb.load_state_dict(ck["ema_combine"])
        dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                            num_patches=256, pretrained_path=None).to(DEVICE).eval()
        dec.load_state_dict(ck["ema_dec"])
        decs[m] = (cb, dec)
        del ck

    col_titles = ["Target (DINOv3 enc.)", f"ours (feed-{args.subset})", f"RAEv2 (feed-{args.subset})"]
    scenes = []
    for spec, label in IMAGES:
        img_np, feats, queries = build(enc, decs, spec, idx, args.block)
        print(f"[{label}] {spec} block {args.block} done", flush=True)
        # per-image: 3 query rows (subjectA / subjectB / background)
        rows = [dict(img_np=img_np, feats=feats, q=q, r=r, c=c, star=STAR_COLORS[k],
                     label=ROW_NAMES[k], label_color=(STAR_COLORS[k] if k < 2 else "0.35"))
                for k, (q, r, c) in enumerate(queries)]
        render(rows, col_titles, f"{OUT_DIR}/repa_{label}_{args.subset}_b{args.block}.png")
        # keep subject-A query for the multi-scene figure
        qa, ra, ca = queries[0]
        scenes.append(dict(img_np=img_np, feats=feats, q=qa, r=ra, c=ca, star="#ff2b2b",
                           label=label, label_color="0.2"))

    render(scenes, col_titles, f"{OUT_DIR}/repa_multiscene_{args.subset}_b{args.block}.png")


if __name__ == "__main__":
    main()
