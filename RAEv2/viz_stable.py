"""Internal-attention stability: ours (random-drop) vs RAEv2 (fixed mean).

Horizontal figure, 2 rows (top = ours, bottom = RAEv2):
    [ input + 3 query ★ ] | [ query A attn ] | [ query B attn ] | [ query C attn ]
Same native input (all 23 layers); patch-level feature-affinity (cosine sim of a
*-marked query patch to every patch) at a chosen decoder block. Aggressive cutoff
so only the most-similar patches glow bright green. Run at a shallow and a middle
block to see how the difference emerges with depth.

  python viz_stable.py --img npz:13150 --block 6  --out assets/viz_pca/stable_corgi_b6.png
  python viz_stable.py --img npz:13150 --block 14 --out assets/viz_pca/stable_corgi_b14.png
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
CUT_LO, CUT_HI = 80, 99.5                           # aggressive: only top ~20% gets color
SUBSET_IDX = {"native": None, "k7": [10, 12, 14, 16, 18, 20, 22], "l11": [10]}
STAR_COLORS = ["#ff2b2b", "#ffd400", "#00d0ff"]     # query A / B / C
# brightness-encoded GREEN colormap (dark -> bright green); brightest = most similar
GREEN = LinearSegmentedColormap.from_list(
    "kgreen", ["#000000", "#04220f", "#0c5e2b", "#1fb04e", "#74f08a", "#e2ffd6"])
MODELS = {
    "ours": dict(ckpt="output_full/decoder_random_drop_layer_mls_plain_cls_k23_16ep/ckpt_ep016.pt",
                 weighting="random_drop", cls_surrogate=True, title="ours (random-drop)", tcol="#1a8f3c"),
    "raev2": dict(ckpt="output_full/decoder_raev2_k23/ckpt_latest.pt",
                  weighting="mean", cls_surrogate=False, title="RAEv2 (fixed mean)", tcol="0.3"),
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
    """patch-level cosine sim of query q to all patches, aggressive cutoff."""
    fn = F.normalize(feat, dim=-1)
    aff = (fn @ fn[q]).reshape(GRID, GRID).numpy()
    lo, hi = np.percentile(aff, CUT_LO), np.percentile(aff, CUT_HI)
    aff = np.clip((aff - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.kron(aff, np.ones((16, 16)))


def pick_queries(target_feat):
    """3 patch points: two on the salient object, one on the background."""
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
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--subset", default="native", choices=list(SUBSET_IDX))
    ap.add_argument("--query", default=None, help="override ra,ca,rb,cb,rg,cg")
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
        queries = [(v[0] * GRID + v[1], v[0], v[1]), (v[2] * GRID + v[3], v[2], v[3]),
                   (v[4] * GRID + v[5], v[4], v[5])]
    else:
        queries = pick_queries(toks[-1][0].float())
    print(f"queries: {[(r, c) for _, r, c in queries]}", flush=True)

    feats = {}
    for m, mc in MODELS.items():
        ck = torch.load(mc["ckpt"], map_location="cpu", weights_only=False)
        cb = MLSCombine(layers=LAYERS, weighting=mc["weighting"], cls_surrogate=mc["cls_surrogate"],
                        projector="none", dim=1024, out_dim=1024).to(DEVICE).eval()
        cb.load_state_dict(ck["ema_combine"])
        dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                            num_patches=256, pretrained_path=None).to(DEVICE).eval()
        dec.load_state_dict(ck["ema_dec"])
        del ck
        with torch.no_grad():
            out = dec(cb(toks, idx=SUBSET_IDX[args.subset]), drop_cls_token=False,
                      output_hidden_states=True)
        feats[m] = out.hidden_states[args.block][0, 1:, :].float()
        print(f"[{m}] block {args.block} done", flush=True)

    gray = np.asarray(Image.fromarray(img_np).convert("L").convert("RGB")) / 255.0
    fig = plt.figure(figsize=(3.0 * 4 + 0.6, 3.0 * 2))
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.05, hspace=0.05)

    axo = fig.add_subplot(gs[:, 0]); axo.imshow(img_np)
    for (q, r, c), col in zip(queries, STAR_COLORS):
        axo.scatter([(c + 0.5) * 16], [(r + 0.5) * 16], marker="*", s=320,
                    c=col, edgecolors="black", linewidths=1.4, zorder=5)
    axo.set_xticks([]); axo.set_yticks([]); axo.set_title(f"input  (block {args.block})", fontsize=12)

    for ri, m in enumerate(["ours", "raev2"]):
        for ci, (q, r, c) in enumerate(queries):
            ax = fig.add_subplot(gs[ri, 1 + ci])
            ax.imshow(0.30 * gray + 0.03)
            ax.imshow(affinity(feats[m], q), cmap=GREEN, alpha=0.92, vmin=0, vmax=1,
                      interpolation="nearest")
            ax.scatter([(c + 0.5) * 16], [(r + 0.5) * 16], marker="*", s=260,
                       c=STAR_COLORS[ci], edgecolors="black", linewidths=1.2, zorder=5)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"query {'ABC'[ci]} ★", fontsize=12, color=STAR_COLORS[ci])
            if ci == 0:
                ax.set_ylabel(MODELS[m]["title"], fontsize=12, color=MODELS[m]["tcol"])
    cax = fig.add_subplot(gs[:, -1])
    fig.colorbar(ScalarMappable(cmap=GREEN), cax=cax)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
