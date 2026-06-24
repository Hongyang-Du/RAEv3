"""16-panel REPA-style strip for ONE image.

  [ original + 3 query ★ ]  +  3 rows (query pts) x 5 cols:
      ours·k23 | ours·k7 | ours·l11 | RAEv2·k7 | RAEv2·l11
  = 1 + 15 = 16 panels, shared viridis colorbar (matches the reference figure).

Same DINOv3 tokens fed with different layer subsets (k23=all 23 / k7 / l11);
deep decoder block; affinity = cosine sim of a *-marked query patch to all
patches (per-map [2,98]-pct stretched), viridis, [0,1] colorbar on the right.

  python viz_strip16.py --img npz:14200 --block 24 --out assets/viz_pca/strip16_cat_b24.png
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
from matplotlib.cm import ScalarMappable

DEVICE = "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LAYERS = list(range(1, 24))
GRID = 16
CMAP = "magma"                                     # warm heat (no green), high brightness contrast
STAR_COLORS = ["#ff2b2b", "#ffd400", "#ffffff"]    # subjectA, subjectB, background
SUBSETS = {"k23": None, "k7": [10, 12, 14, 16, 18, 20, 22], "l11": [10]}
MODELS = {
    "ours": dict(ckpt="output_full/decoder_random_drop_layer_mls_plain_cls_k23_16ep/ckpt_ep016.pt",
                 weighting="random_drop", cls_surrogate=True),
    "raev2": dict(ckpt="output_full/decoder_raev2_k23/ckpt_latest.pt",
                  weighting="mean", cls_surrogate=False),
}
# 5 columns: (title, model, subset, title_color)
COLUMNS = [("ours · k23", "ours", "k23", "#d2691e"),
           ("ours · k7", "ours", "k7", "#d2691e"),
           ("ours · l11", "ours", "l11", "#d2691e"),
           ("RAEv2 · k7", "raev2", "k7", "0.3"),
           ("RAEv2 · l11", "raev2", "l11", "0.3")]

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
    """cosine sim of query q to all patches at PATCH resolution (16x16 blocks,
    no per-pixel interpolation), per-map [2,98]-pct stretched, tiled to 256."""
    fn = F.normalize(feat, dim=-1)
    aff = (fn @ fn[q]).reshape(GRID, GRID).numpy()
    lo, hi = np.percentile(aff, 50), np.percentile(aff, 99)   # darken bottom half -> bright blob pops
    aff = np.clip((aff - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.kron(aff, np.ones((16, 16)))         # blocky 256x256 (patch level)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--block", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    x, img_np = load_image(args.img)
    enc = create_encoder("dinov3mls-vit-l16[layers=" + ".".join(map(str, LAYERS)) + "]",
                         device=DEVICE, resolution=256).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    toks = list(enc.model.get_intermediate_layers(
        (x - MEAN) / STD, n=LAYERS, reshape=False, return_class_token=False, norm=True))
    queries = pick_queries(toks[-1][0].float())
    print(f"queries: {[(r, c) for _, r, c in queries]}", flush=True)

    # feats[model][subset] = block-`block` patch features for the needed (model,subset) pairs
    need = {("ours", "k23"), ("ours", "k7"), ("ours", "l11"), ("raev2", "k7"), ("raev2", "l11")}
    feats = {}
    for m, mc in MODELS.items():
        if not any(mm == m for mm, _ in need):
            continue
        ck = torch.load(mc["ckpt"], map_location="cpu", weights_only=False)
        cb = MLSCombine(layers=LAYERS, weighting=mc["weighting"], cls_surrogate=mc["cls_surrogate"],
                        projector="none", dim=1024, out_dim=1024).to(DEVICE).eval()
        cb.load_state_dict(ck["ema_combine"])
        dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                            num_patches=256, pretrained_path=None).to(DEVICE).eval()
        dec.load_state_dict(ck["ema_dec"])
        del ck
        for s in ["k23", "k7", "l11"]:
            if (m, s) not in need:
                continue
            with torch.no_grad():
                out = dec(cb(toks, idx=SUBSETS[s]), drop_cls_token=False, output_hidden_states=True)
            feats[(m, s)] = out.hidden_states[args.block][0, 1:, :].float()
        print(f"[{m}] block {args.block} done", flush=True)

    # ---- figure: original (left, centered) + 3x5 grid + colorbar ----
    gray = np.asarray(Image.fromarray(img_np).convert("L").convert("RGB")) / 255.0
    fig = plt.figure(figsize=(2.55 * (1 + len(COLUMNS)) + 0.6, 2.55 * 3))
    gs = GridSpec(3, 1 + len(COLUMNS) + 1, figure=fig,
                  width_ratios=[1] + [1] * len(COLUMNS) + [0.05], wspace=0.05, hspace=0.05)

    ax0 = fig.add_subplot(gs[1, 0]); ax0.imshow(img_np)
    for (q, r, c), col in zip(queries, STAR_COLORS):
        ax0.scatter([(c + 0.5) * 16], [(r + 0.5) * 16], marker="*", s=300,
                    c=col, edgecolors="black", linewidths=1.3, zorder=5)
    ax0.set_xticks([]); ax0.set_yticks([]); ax0.set_title("input", fontsize=12)

    for ri, (q, r, c) in enumerate(queries):
        for ci, (title, m, s, tcol) in enumerate(COLUMNS):
            ax = fig.add_subplot(gs[ri, 1 + ci])
            ax.imshow(0.30 * gray + 0.03)
            ax.imshow(affinity(feats[(m, s)], q), cmap=CMAP, alpha=0.9, vmin=0, vmax=1,
                      interpolation="nearest")
            ax.scatter([(c + 0.5) * 16], [(r + 0.5) * 16], marker="*", s=260,
                       c=STAR_COLORS[ri], edgecolors="black", linewidths=1.2, zorder=5)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(title, fontsize=12, color=tcol)

    cax = fig.add_subplot(gs[:, -1])
    fig.colorbar(ScalarMappable(cmap=CMAP), cax=cax)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
