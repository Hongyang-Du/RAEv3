"""Combined robustness figure (clean):

  LEFT  = single-image vis (rows = ours / raev2):
            [ input + query star ] [ affinity heatmap @ b14 ] [ @ b28 ] [ recon ]
          REPA-style feature-affinity heatmaps: cosine sim of the *-marked query
          patch's feature to every patch, bicubic-upsampled & overlaid (smooth,
          unlike low-res PCA). Computed on the SUBSET-fed features.
  RIGHT = stats curve over 1000 val images: per-block feature error (relL2 vs
          native), ours=orange / raev2=gray, mean +/- std.

  python viz_combined.py --img npz:14601 --subset k7 --label tiger \
      --stats assets/viz_pca/stats_1000.npz --out assets/viz_pca/combined_tiger_k7.png
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

from encoders.vision_encoder import create_encoder
from stage1.rae import _load_decoder
from stage1.combine import MLSCombine

DEVICE = "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LAYERS = list(range(1, 24))
GRID = 16
ALL_BLOCKS = list(range(0, 29))
MID_BLOCK = 14                                     # only the middle decoder block
STAR_COLORS = ["#ff2b2b", "#ffd400", "#ffffff"]    # subjectA, subjectB, background
# blue-green + brightness colormap (no rainbow), REPA fig.6 style
CMAP = LinearSegmentedColormap.from_list(
    "bluegreen", ["#06101c", "#0b3b54", "#10746e", "#3fbf8f", "#d6f5cf"])
SUBSET_IDX = {"k7": [10, 12, 14, 16, 18, 20, 22], "L11": [10]}
MODELS = {
    "ours": dict(ckpt="output_full/decoder_random_drop_layer_mls_plain_cls_k23_16ep/ckpt_ep016.pt",
                 weighting="random_drop", cls_surrogate=True, color="#ff7f0e"),
    "raev2": dict(ckpt="output_full/decoder_raev2_k23/ckpt_latest.pt",
                  weighting="mean", cls_surrogate=False, color="0.45"),
}
TITLE = {"ours": "ours (random-drop+cls)", "raev2": "RAEv2 k23 (fixed mean)"}


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
def forward_all(dec, combine, toks, idx):
    z = combine(toks, idx=idx)
    out = dec(z, drop_cls_token=False, output_hidden_states=True)
    rec = (dec.unpatchify(out.logits) * STD + MEAN).clamp(0, 1)
    rec = rec[0].permute(1, 2, 0).mul(255).round().byte().numpy()
    feats = {b: out.hidden_states[b][0, 1:, :].float() for b in ALL_BLOCKS}
    return rec, feats


def affinity_map(feat, qidx):
    """cosine sim of query patch feature to all patches -> 256x256 bicubic,
    then per-map [2,98]-percentile contrast stretch to [0,1] for display."""
    fn = F.normalize(feat, dim=-1)                 # [256, D]
    aff = (fn @ fn[qidx]).reshape(1, 1, GRID, GRID)  # [1,1,16,16] in [-1,1]
    aff = F.interpolate(aff, size=(256, 256), mode="bicubic", align_corners=False)[0, 0].numpy()
    lo, hi = np.percentile(aff, 2), np.percentile(aff, 98)
    return np.clip((aff - lo) / max(hi - lo, 1e-6), 0, 1)


def put_star(ax, r, c, color):
    ax.scatter([(c + 0.5) * 16], [(r + 0.5) * 16], marker="*", s=300,
               c=color, edgecolors="black", linewidths=1.3, zorder=5)


def overlay(ax, img_np, heat):
    gray = np.asarray(Image.fromarray(img_np).convert("L").convert("RGB")) / 255.0
    ax.imshow(0.40 * gray + 0.04)                  # dim background
    ax.imshow(heat, cmap=CMAP, alpha=0.72, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--subset", required=True, choices=list(SUBSET_IDX))
    ap.add_argument("--label", default=None)
    ap.add_argument("--stats", default="assets/viz_pca/stats_1000.npz")
    ap.add_argument("--query", default=None, help="row,col patch (0-15); default auto-salient")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    idx = SUBSET_IDX[args.subset]
    label = args.label or os.path.splitext(os.path.basename(args.img))[0]

    x, img_np = load_image(args.img)
    enc = create_encoder("dinov3mls-vit-l16[layers=" + ".".join(map(str, LAYERS)) + "]",
                         device=DEVICE, resolution=256).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    toks = list(enc.model.get_intermediate_layers(
        (x - MEAN) / STD, n=LAYERS, reshape=False, return_class_token=False, norm=True))

    res = {}
    for m, mc in MODELS.items():
        ck = torch.load(mc["ckpt"], map_location="cpu", weights_only=False)
        cb = MLSCombine(layers=LAYERS, weighting=mc["weighting"], cls_surrogate=mc["cls_surrogate"],
                        projector="none", dim=1024, out_dim=1024).to(DEVICE).eval()
        cb.load_state_dict(ck["ema_combine"])
        dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                            num_patches=256, pretrained_path=None).to(DEVICE).eval()
        dec.load_state_dict(ck["ema_dec"])
        del ck
        _, nat = forward_all(dec, cb, toks, None)
        rec, sub = forward_all(dec, cb, toks, idx)
        res[m] = dict(rec=rec, sub=sub, nat=nat)
        print(f"[{m}] done", flush=True)

    # three query patches (shared by both rows), from ours native final-block saliency:
    #   subjectA = center-biased argmax; subjectB = next salient peak away from A;
    #   background = low-saliency point far from both.
    norm = res["ours"]["nat"][28].norm(dim=-1).reshape(GRID, GRID).numpy()
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    gw = np.exp(-(((yy - 7.5) ** 2 + (xx - 7.5) ** 2) / (2 * 4.5 ** 2)))
    sal = norm * gw
    qa = int(sal.argmax()); ra, ca = qa // GRID, qa % GRID
    s2 = sal.copy(); s2[(np.abs(yy - ra) + np.abs(xx - ca)) < 4] = -np.inf
    qb = int(s2.argmax()); rb, cb = qb // GRID, qb % GRID
    farboth = ((np.abs(yy - ra) + np.abs(xx - ca)) >= 5) & ((np.abs(yy - rb) + np.abs(xx - cb)) >= 5)
    sc = norm.copy(); sc[~farboth] = np.inf
    qg = int(sc.argmin()); rg, cg = qg // GRID, qg % GRID
    if args.query:                                           # override "ra,ca,rb,cb,rg,cg"
        ra, ca, rb, cb, rg, cg = (int(v) for v in args.query.split(","))
        qa, qb, qg = ra * GRID + ca, rb * GRID + cb, rg * GRID + cg
    queries = [(qa, ra, ca), (qb, rb, cb), (qg, rg, cg)]      # subjectA, subjectB, background
    print(f"queries (r,c): A=({ra},{ca}) B=({rb},{cb}) bg=({rg},{cg})", flush=True)

    # per-block feature error (relL2 vs native) for THIS image
    def curve(m):
        nat, sub = res[m]["nat"], res[m]["sub"]
        return [((sub[b] - nat[b]).norm(dim=-1) /
                 nat[b].norm(dim=-1).clamp_min(1e-6)).mean().item() for b in ALL_BLOCKS]

    # ---- figure: [input+3★] [affA] [affB] [affBg] [recon] | curve ----
    fig = plt.figure(figsize=(17.5, 6.0))
    gs = GridSpec(2, 7, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.10, 2.4],
                  hspace=0.08, wspace=0.06)
    titles = ["input", "subject A", "subject B", "background", "recon"]
    for ri, m in enumerate(["ours", "raev2"]):
        ax = fig.add_subplot(gs[ri, 0]); ax.imshow(img_np)
        for (q, r, c), col in zip(queries, STAR_COLORS):
            put_star(ax, r, c, col)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(TITLE[m], fontsize=11,
                      color=MODELS[m]["color"] if m == "ours" else "0.25")
        if ri == 0:
            ax.set_title(titles[0], fontsize=10)
        for ci, ((q, r, c), col) in enumerate(zip(queries, STAR_COLORS)):
            ax = fig.add_subplot(gs[ri, 1 + ci])
            overlay(ax, img_np, affinity_map(res[m]["sub"][MID_BLOCK], q))
            put_star(ax, r, c, col)
            if ri == 0:
                ax.set_title(titles[1 + ci], fontsize=10)
        ax = fig.add_subplot(gs[ri, 4]); ax.imshow(res[m]["rec"])
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title(titles[4], fontsize=10)

    # single-image error curve (y-axis on the right, no title/legend)
    axc = fig.add_subplot(gs[:, 6])
    for m in ["ours", "raev2"]:
        axc.plot(ALL_BLOCKS, curve(m), marker="o", ms=3.5, lw=2.0, color=MODELS[m]["color"])
    axc.set_xlabel("decoder block (0=embed .. 28)")
    axc.set_ylabel("feature error vs native  (relative L2, lower = robust)")
    axc.yaxis.set_label_position("right"); axc.yaxis.tick_right()
    axc.grid(alpha=0.3); axc.set_ylim(bottom=0)

    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
