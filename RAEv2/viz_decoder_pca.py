"""Decoder intermediate-feature PCA->RGB robustness visualization.

For one image we feed the SAME trained decoder different DINOv3 input-layer
subsets (native 23 / feed-k7 / feed-L11) and PCA the decoder's intermediate
block features to RGB. To make color drift *meaningful*, the PCA basis +
color-normalization for a given (model, block) are fit ONCE on the native
subset and the other subsets are projected through that same basis. Stable
color across subsets == representation robust to which input layers are fed.

We compare OUR random-drop+cls k23 decoder vs the RAEv2 fixed-mean k23 decoder.
Both live in this codebase and differ only in the combine weighting.

Runs on CPU (single image) to avoid disturbing the running training.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from encoders.vision_encoder import create_encoder
from stage1.rae import _load_decoder
from stage1.combine import MLSCombine

DEVICE = "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LAYERS = list(range(1, 24))                      # DINOv3 layers 1..23
GRID = 16                                         # 256/16 patches per side
ALL_BLOCKS = list(range(0, 29))                  # every decoder hidden_states index (0=embed..28)
BLOCKS = [2, 6, 10, 14, 18, 22, 26, 28]          # denser set shown in the PCA grid

IMG_PATH = "assets/samples/cat.jpg"
OUT_DIR = "assets/viz_pca"

# eval idx (0-based positions into LAYERS) for each input-layer subset
SUBSETS = {
    "native (23)": None,
    "feed-k7": [10, 12, 14, 16, 18, 20, 22],     # layers 11,13,15,17,19,21,23
    "feed-L11": [10],                            # layer 11 only
}

MODELS = {
    "ours (random-drop+cls)": dict(
        ckpt="output_full/decoder_random_drop_layer_mls_plain_cls_k23_16ep_gan8/ckpt_latest.pt",
        weighting="random_drop", cls_surrogate=True),
    "RAEv2 k23 (fixed mean)": dict(
        ckpt="output_full/decoder_raev2_k23/ckpt_latest.pt",
        weighting="mean", cls_surrogate=False),
}


def load_image():
    img = Image.open(IMG_PATH).convert("RGB").resize((256, 256), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0), np.asarray(img)


@torch.no_grad()
def encode_tokens(enc, x):
    """list of 23 [1, 256, 1024] frozen DINOv3 per-layer patch tokens."""
    return list(enc.model.get_intermediate_layers(
        (x - MEAN) / STD, n=LAYERS, reshape=False,
        return_class_token=False, norm=True))


@torch.no_grad()
def decoder_feats(dec, combine, toks, idx):
    """returns (recon[256,256,3] uint8, {block: feat[256,1152]})."""
    z = combine(toks, idx=idx)
    out = dec(z, drop_cls_token=False, output_hidden_states=True)
    rec = (dec.unpatchify(out.logits) * STD + MEAN).clamp(0, 1)
    rec = rec[0].permute(1, 2, 0).mul(255).round().byte().numpy()
    feats = {b: out.hidden_states[b][0, 1:, :].float() for b in ALL_BLOCKS}  # drop cls token
    return rec, feats


def fit_pca_rgb(native_feat):
    """Fit 3-comp PCA + 1/99-pct color scale on native features. Returns a
    projector that maps any [256,1152] feat -> [16,16,3] RGB in [0,1] using the
    SAME basis/scale, so cross-subset color drift reflects feature drift."""
    mu = native_feat.mean(0)
    Xc = native_feat - mu
    _, _, V = torch.linalg.svd(Xc, full_matrices=False)
    comps = V[:3]                                # [3, 1152]
    proj_native = Xc @ comps.T                   # [256, 3]
    lo = torch.quantile(proj_native, 0.01, dim=0)
    hi = torch.quantile(proj_native, 0.99, dim=0)
    rng = (hi - lo).clamp_min(1e-6)

    def project(feat):
        p = (feat - mu) @ comps.T
        p = ((p - lo) / rng).clamp(0, 1)
        return p.reshape(GRID, GRID, 3).numpy()

    return project


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(42)

    x, img_np = load_image()
    enc = create_encoder("dinov3mls-vit-l16[layers=" + ".".join(map(str, LAYERS)) + "]",
                         device=DEVICE, resolution=256).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    toks = encode_tokens(enc, x)
    print(f"[encode] {len(toks)} layers, token shape {tuple(toks[0].shape)}", flush=True)

    # per model: recon + intermediate feats per subset, and a native-fit PCA per block
    results = {}
    for mname, mc in MODELS.items():
        print(f"[model] {mname}: {mc['ckpt']}", flush=True)
        ck = torch.load(mc["ckpt"], map_location="cpu", weights_only=False)
        combine = MLSCombine(layers=LAYERS, weighting=mc["weighting"],
                             cls_surrogate=mc["cls_surrogate"], projector="none",
                             dim=1024, out_dim=1024).to(DEVICE).eval()
        combine.load_state_dict(ck["ema_combine"])
        dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                            num_patches=256, pretrained_path=None).to(DEVICE).eval()
        dec.load_state_dict(ck["ema_dec"])
        del ck

        per_subset = {}
        for sname, idx in SUBSETS.items():
            rec, feats = decoder_feats(dec, combine, toks, idx)
            per_subset[sname] = dict(rec=rec, feats=feats)
            print(f"   {sname}: ok", flush=True)
        # PCA basis fit on native, per block (grid uses BLOCKS subset)
        projectors = {b: fit_pca_rgb(per_subset["native (23)"]["feats"][b]) for b in BLOCKS}
        # drift vs native per (subset, block) for ALL blocks: cosine (direction)
        # AND relative-L2 (magnitude-aware) — the latter tracks the recon collapse.
        nat = per_subset["native (23)"]["feats"]
        drift, rell2 = {}, {}
        for sname in SUBSETS:
            drift[sname], rell2[sname] = {}, {}
            for b in ALL_BLOCKS:
                f, n = per_subset[sname]["feats"][b], nat[b]
                drift[sname][b] = torch.cosine_similarity(f, n, dim=-1).mean().item()
                rell2[sname][b] = ((f - n).norm(dim=-1) /
                                   n.norm(dim=-1).clamp_min(1e-6)).mean().item()
        results[mname] = dict(sub=per_subset, proj=projectors, drift=drift, rell2=rell2)

    # ---- figure: rows = [recon] + BLOCKS ; cols = models x subsets ----
    sub_names = list(SUBSETS.keys())
    model_names = list(MODELS.keys())
    ncol = len(model_names) * len(sub_names)
    nrow = 1 + len(BLOCKS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.35 * nrow))

    def col_idx(mi, si):
        return mi * len(sub_names) + si

    for mi, mname in enumerate(model_names):
        for si, sname in enumerate(sub_names):
            c = col_idx(mi, si)
            r = results[mname]
            # row 0: reconstruction
            ax = axes[0, c]
            ax.imshow(r["sub"][sname]["rec"])
            ax.set_xticks([]); ax.set_yticks([])
            if si == 0:
                ax.set_ylabel("recon", fontsize=10)
            ax.set_title(sname, fontsize=9)
            # block rows: PCA RGB
            for ri, b in enumerate(BLOCKS):
                ax = axes[ri + 1, c]
                rgb = r["proj"][b](r["sub"][sname]["feats"][b])
                rgb = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize(
                    (256, 256), Image.NEAREST))
                ax.imshow(rgb)
                ax.set_xticks([]); ax.set_yticks([])
                cos = r["drift"][sname][b]
                ax.set_title(f"cos={cos:.3f}", fontsize=8,
                             color=("black" if si == 0 else ("green" if cos > 0.9 else "red")))
                if c == 0:
                    ax.set_ylabel(f"block {b}", fontsize=10)
        # group header spanning this model's columns
        x0 = (col_idx(mi, 0) + 0.0) / ncol
        x1 = (col_idx(mi, len(sub_names) - 1) + 1.0) / ncol
        fig.text((x0 + x1) / 2, 0.995, mname, ha="center", va="top",
                 fontsize=12, fontweight="bold")

    fig.suptitle("", y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.975))
    out = os.path.join(OUT_DIR, "decoder_pca_robust_cat.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved -> {out}", flush=True)

    # ---- line plot: drift vs native, layer by layer (2 metrics x subsets) ----
    drift_subs = [s for s in sub_names if s != "native (23)"]
    metrics = [("drift", "mean cosine sim vs native (higher=robust)", (0.5, 1.005), "lower left"),
               ("rell2", "mean relative L2 vs native (lower=robust)", None, "upper left")]
    fig2, axes2 = plt.subplots(len(metrics), len(drift_subs),
                               figsize=(5.2 * len(drift_subs), 3.9 * len(metrics)),
                               squeeze=False)
    colors = {"ours (random-drop+cls)": "C2", "RAEv2 k23 (fixed mean)": "C3"}
    for i, (key, ylab, ylim, legloc) in enumerate(metrics):
        for j, sname in enumerate(drift_subs):
            ax = axes2[i, j]
            for mname in model_names:
                ys = [results[mname][key][sname][b] for b in ALL_BLOCKS]
                ax.plot(ALL_BLOCKS, ys, marker="o", ms=3, lw=1.8,
                        color=colors.get(mname), label=mname)
            ax.set_title(f"{sname} vs native", fontsize=11)
            ax.set_xlabel("decoder block (0=embed .. 28)")
            ax.grid(alpha=0.3)
            if ylim:
                ax.set_ylim(*ylim)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=10)
            ax.legend(fontsize=9, loc=legloc)
    fig2.suptitle("Intermediate-feature drift propagation through decoder blocks",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "decoder_pca_drift_propagation_cat.png")
    fig2.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"saved -> {out2}", flush=True)

    # print drift tables
    for key, title in [("drift", "cosine sim (higher=robust)"),
                       ("rell2", "relative L2 (lower=robust)")]:
        print(f"\n=== {title} of intermediate feats vs native ===")
        for mname in model_names:
            print(f"\n{mname}")
            print("  block " + "".join(f"{s:>16}" for s in sub_names))
            for b in BLOCKS:
                print(f"  {b:>5} " + "".join(
                    f"{results[mname][key][s][b]:>16.4f}" for s in sub_names))


if __name__ == "__main__":
    main()
