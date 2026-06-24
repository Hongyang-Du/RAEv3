"""Grouped-by-input-layer patch-query attention grid.

2 rows (top = ours random-drop, bottom = RAEv2 fixed-mean). Each row is grouped
by INPUT LAYER subset (k23 / k7 / l11); each group = [ input + 3 query ★ ] plus
the 3 query patches' attention maps at a chosen decoder block. Green colormap,
aggressive cutoff (only the most-similar patches glow). Run several blocks to
pick the best one for the visualization.

  python viz_grid.py --img npz:10350 --block 14 --out assets/viz_pca/grid_dog_b14.png
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
CMAP = "viridis"                                     # match the reference figure
STAR_COLORS = ["#ff2b2b", "#ffd400", "#00d0ff"]      # query A / B / C
GROUPS = [("k23", None, "input layers: k = 23 (all layers)"),
          ("k7", [10, 12, 14, 16, 18, 20, 22], "input layers: k = 7 (11,..,23)"),
          ("l11", [10], "input layer: l = 11 (single layers)")]
MODELS = {
    "ours": dict(ckpt="output_full/decoder_random_drop_layer_mls_plain_cls_k23_16ep/ckpt_ep016.pt",
                 weighting="random_drop", cls_surrogate=True, title="OmniRAE (ours)", tcol="black"),
    "raev2": dict(ckpt="output_full/decoder_raev2_k23/ckpt_latest.pt",
                  weighting="mean", cls_surrogate=False, title="RAEv2 K=23", tcol="0.3"),
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
    """patch-level cosine sim of query q to all patches, per-map min-max -> 256x256."""
    fn = F.normalize(feat, dim=-1)
    aff = (fn @ fn[q]).reshape(GRID, GRID).numpy()
    aff = (aff - aff.min()) / (aff.max() - aff.min() + 1e-6)
    return np.kron(aff, np.ones((16, 16)))


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


def star(ax, r, c, col):
    ax.scatter([(c + 0.5) * 16], [(r + 0.5) * 16], marker="*", s=120,
               c=col, edgecolors="black", linewidths=1.0, zorder=5)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--query", default=None, help="manual patches: ra,ca,rb,cb,rg,cg (row,col 0-15)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    x, img_np = load_image(args.img)
    enc = create_encoder("dinov3mls-vit-l16[layers=" + ".".join(map(str, LAYERS)) + "]",
                         device=DEVICE, resolution=256).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    toks = list(enc.model.get_intermediate_layers(
        (x - MEAN) / STD, n=LAYERS, reshape=False, return_class_token=False, norm=True))
    if args.query:
        v = [int(t) for t in args.query.split(",")]
        queries = [(v[2 * k] * GRID + v[2 * k + 1], v[2 * k], v[2 * k + 1]) for k in range(3)]
    else:
        queries = pick_queries(toks[-1][0].float())
    print(f"queries: {[(r, c) for _, r, c in queries]}", flush=True)

    feats, recons = {}, {}                              # feats[model][gname], recons[model][gname]
    for m, mc in MODELS.items():
        ck = torch.load(mc["ckpt"], map_location="cpu", weights_only=False)
        cb = MLSCombine(layers=LAYERS, weighting=mc["weighting"], cls_surrogate=mc["cls_surrogate"],
                        projector="none", dim=1024, out_dim=1024).to(DEVICE).eval()
        cb.load_state_dict(ck["ema_combine"])
        dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                            num_patches=256, pretrained_path=None).to(DEVICE).eval()
        dec.load_state_dict(ck["ema_dec"])
        del ck
        feats[m] = {}; recons[m] = {}
        for gname, gidx, _cap in GROUPS:
            with torch.no_grad():
                out = dec(cb(toks, idx=gidx), drop_cls_token=False, output_hidden_states=True)
            feats[m][gname] = out.hidden_states[args.block][0, 1:, :].float()
            rec = (dec.unpatchify(out.logits) * STD + MEAN).clamp(0, 1)
            recons[m][gname] = rec[0].permute(1, 2, 0).numpy()
        print(f"[{m}] block {args.block} done", flush=True)

    # layout: standalone input column (left) + 3 groups, each [3 queries + recon]
    # glued; minimal gap between input/groups.
    gray = np.asarray(Image.fromarray(img_np).convert("L").convert("RGB")) / 255.0
    GAP = 0.08                                          # minimal but visible margin
    wr = [1, GAP] + ([1, 1, 1, 1, GAP] * 3) + [0.05]    # input | g1 | g2 | g3 | cbar
    fig = plt.figure(figsize=(1.85 * 14, 1.85 * 2))
    gs = GridSpec(2, len(wr), figure=fig, width_ratios=wr, wspace=0.0, hspace=0.0)

    for ri, m in enumerate(["ours", "raev2"]):
        axi = fig.add_subplot(gs[ri, 0]); axi.imshow(img_np, aspect="auto")
        for (q, r, c), scol in zip(queries, STAR_COLORS):
            star(axi, r, c, scol)
        axi.set_ylabel(MODELS[m]["title"], fontsize=12, color=MODELS[m]["tcol"])
        if ri == 0:
            axi.set_title("input", fontsize=12)
        for gi, (gname, _gidx, cap) in enumerate(GROUPS):
            c0 = 2 + gi * 5                             # group start column
            for k, (q, r, c) in enumerate(queries):
                ax = fig.add_subplot(gs[ri, c0 + k])
                ax.imshow(0.42 * gray + 0.04, aspect="auto")
                ax.imshow(affinity(feats[m][gname], q), cmap=CMAP, alpha=0.82,
                          vmin=0, vmax=1, interpolation="nearest", aspect="auto")
                star(ax, r, c, STAR_COLORS[k])
                if ri == 0 and k == 0:
                    ax.set_title(cap, fontsize=12, loc="left")
            ax = fig.add_subplot(gs[ri, c0 + 3]); ax.imshow(recons[m][gname], aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])

    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(ScalarMappable(cmap=CMAP), cax=cax)
    cb.set_label("cosine similarity", fontsize=12, rotation=90)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    pdf = args.out.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf, bbox_inches="tight")
    print(f"saved -> {args.out}  &  {pdf}", flush=True)


if __name__ == "__main__":
    main()
