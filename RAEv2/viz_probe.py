"""Linear-probe semantic content across encoder THEN decoder layers.

One figure: DINOv3 encoder layers 1..23 (shared curve) followed by decoder
blocks 0..28 for OmniRAE (ours) vs RAEv2 (two curves that fork in the decoder).

Labels come from the class-sorted val npz (50/class -> label = idx//50); per class
the first 40 images are probe-train, the last 10 are held-out top-1 eval. Feature
= patch-token mean-pool; head = standardize + Linear(d, 1000). Decoder fed native.

  python viz_probe.py --out assets/viz_pca/probe_enc_dec.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LAYERS = list(range(1, 24))
NB = 29
MODELS = {
    "ours": dict(ckpt="output_full/decoder_random_drop_layer_mls_plain_cls_k23_16ep/ckpt_ep016.pt",
                 weighting="random_drop", cls_surrogate=True, title="OmniRAE (ours)", color="#e8770c"),
    "raev2": dict(ckpt="output_full/decoder_raev2_k23/ckpt_latest.pt",
                  weighting="mean", cls_surrogate=False, title="RAEv2 K=23", color="0.4"),
}

from stage1.rae import _load_decoder                       # noqa: E402
from stage1.combine import MLSCombine                      # noqa: E402
from encoders.vision_encoder import create_encoder         # noqa: E402


@torch.no_grad()
def extract(args, dev):
    arr = np.load("data_eval/imagenet-256-val.npz", mmap_mode="r")["arr_0"]
    N = len(arr)                                           # 50000, class-sorted 50/class
    y = torch.arange(N) // 50
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

    Fe = [torch.empty(N, 1024, dtype=torch.float16) for _ in LAYERS]
    Fd = {m: [torch.empty(N, 1152, dtype=torch.float16) for _ in range(NB)] for m in MODELS}
    for i in range(0, N, args.batch):
        j = min(i + args.batch, N)
        imgs = torch.stack([torch.from_numpy(arr[k].copy()) for k in range(i, j)])
        imgs = imgs.permute(0, 3, 1, 2).float().to(dev) / 255
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inter = list(enc.model.get_intermediate_layers(
                (imgs - mean) / std, n=LAYERS, reshape=False,
                return_class_token=True, norm=True))
            patch = [o[0] for o in inter]                 # patch tokens fed to the decoder
            for li in range(len(LAYERS)):
                Fe[li][i:j] = inter[li][1].half().cpu()    # encoder CLS token
            for m, (cb, dec) in decs.items():
                out = dec(cb(patch, idx=None), drop_cls_token=False, output_hidden_states=True)
                for b in range(NB):
                    Fd[m][b][i:j] = out.hidden_states[b][:, 0, :].half().cpu()  # decoder CLS token
        if (j) % 2000 == 0:
            print(f"  extract {j}/{N}", flush=True)
    return Fe, Fd, y


def probe(X, y, dev, epochs=30):
    """standardize + Linear(d,1000); train on first-40/class, eval last-10/class."""
    tr = (torch.arange(len(y)) % 50) < 40
    va = ~tr
    Xtr, ytr = X[tr].float(), y[tr]
    Xva, yva = X[va].float(), y[va]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = ((Xtr - mu) / sd).to(dev)
    Xva = ((Xva - mu) / sd).to(dev)
    ytr = ytr.to(dev); yva = yva.to(dev)
    head = nn.Linear(X.shape[1], 1000).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    n = Xtr.shape[0]; bs = 4096
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(head(Xtr[idx]), ytr[idx])
            loss.backward(); opt.step()
    with torch.no_grad():
        acc = (head(Xva).argmax(1) == yva).float().mean().item() * 100
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="assets/viz_pca/probe_enc_dec.png")
    args = ap.parse_args()
    dev = "cuda"

    print("extracting features...", flush=True)
    Fe, Fd, y = extract(args, dev)
    print("training probes...", flush=True)
    enc_acc = [probe(Fe[li], y, dev, args.epochs) for li in range(len(LAYERS))]
    print(f"  encoder done: {[round(a,1) for a in enc_acc]}", flush=True)
    dec_acc = {}
    for m in MODELS:
        dec_acc[m] = [probe(Fd[m][b], y, dev, args.epochs) for b in range(NB)]
        print(f"  {m} decoder done: {[round(a,1) for a in dec_acc[m]]}", flush=True)

    import json
    json.dump({"enc": enc_acc, "dec_ours": dec_acc["ours"], "dec_raev2": dec_acc["raev2"]},
              open(args.out.rsplit(".", 1)[0] + ".json", "w"), indent=2)

    # ---- one figure: encoder curve (shared) then forked decoder curves ----
    xe = list(range(1, len(LAYERS) + 1))                  # 1..23
    GAPX = 2
    xd = [len(LAYERS) + GAPX + b for b in range(NB)]      # decoder 0..28 shifted right
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.plot(xe, enc_acc, "o-", color="#6a3d9a", lw=2.2, ms=5, label="DINOv3 encoder (shared)")
    # connect encoder end to each decoder start
    for m in MODELS:
        ax.plot([xe[-1], xd[0]], [enc_acc[-1], dec_acc[m][0]], "-", color=MODELS[m]["color"], lw=1.2, alpha=0.5)
        ax.plot(xd, dec_acc[m], "o-", color=MODELS[m]["color"], lw=2.2, ms=4, label=MODELS[m]["title"])
    ax.axvspan(0.3, len(LAYERS) + 0.5, color="#6a3d9a", alpha=0.05)
    ax.axvline(len(LAYERS) + GAPX / 2, color="0.6", ls="--", lw=1)
    ax.text(len(LAYERS) / 2 + 0.5, ax.get_ylim()[1], "encoder", ha="center", va="top", fontsize=12, color="#6a3d9a")
    ax.text(len(LAYERS) + GAPX + NB / 2, ax.get_ylim()[1], "decoder", ha="center", va="top", fontsize=12, color="0.3")
    xticks = xe[::2] + xd[::4]
    xlab = [f"L{l}" for l in LAYERS[::2]] + [f"D{b}" for b in range(0, NB, 4)]
    ax.set_xticks(xticks); ax.set_xticklabels(xlab, fontsize=8)
    ax.set_xlabel("layer  (DINOv3 encoder L1..L23  →  decoder block D0..D28)")
    ax.set_ylabel("ImageNet val top-1 [%]  (linear probe)")
    ax.set_title("Semantic content through encoder then decoder (CLS token, linear probe, 40/10 per class)")
    ax.legend(fontsize=10, loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    fig.savefig(args.out.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
