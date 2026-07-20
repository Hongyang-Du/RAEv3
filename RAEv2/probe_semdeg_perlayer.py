"""Semantic-degeneration linear probe — PER-LAYER independent heads.

Hybrid of viz_probe.py (each layer gets its OWN standardize+Linear probe, trained
separately) and the branch semantic-degeneration figure (encoder probed only on the
k7 layers, decoder probed every 2 blocks, three decoder curves, in-distribution feed).

  encoder  : DINOv3-L cls token at layers 11,13,15,17,19,21,23  (semantics RISE)
  dec_ours : our k23 random-drop+cls decoder   <- 23-layer combine + L23 cls surrogate
  dec_k7   : official RAEv2 dinov3l-k7 decoder  <- 7-layer  combine + L23 cls surrogate
  dec_k23  : official RAEv2 dinov3l-k23 decoder <- 23-layer combine + L23 cls surrogate
             (decoder cls token read every 2 transformer blocks -> semantics FALL)

Every decoder is fed the latent IT was trained on (NO off-distribution feed). Features
are cached once, then each (layer) trains an independent linear probe on the class-sorted
val npz: 40 imgs/class train, 10 imgs/class held-out top-1. Output schema matches the
branch plot_semantic_degeneration.py.

  python probe_semdeg_perlayer.py --ours-ckpt output_full/gdrive_omnirae/omnirae_ckpt.pt
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch
import torch.nn as nn

from stage1.rae import _load_decoder                       # noqa: E402
from stage1.combine import MLSCombine                      # noqa: E402
from encoders.vision_encoder import create_encoder         # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

K23_LAYERS = list(range(1, 24))                            # 1..23 (encoder feed for k23/ours)
K7_LAYERS = [11, 13, 15, 17, 19, 21, 23]                  # official k7 feed
ENC_PROBE_LAYERS = [11, 13, 15, 17, 19, 21, 23]           # encoder cls probed only on k7 layers
DEC_BLOCK_STRIDE = 2                                      # probe decoder cls every 2 blocks


def load_decoder_from_ckpt(path, device):
    dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                        num_patches=256, pretrained_path=None)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj["ema_dec"] if isinstance(obj, dict) and "ema_dec" in obj else obj
    missing, unexpected = dec.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [warn] {os.path.basename(path)}: missing={len(missing)} unexpected={len(unexpected)}")
    return dec.to(device).eval()


@torch.no_grad()
def decoder_cls_per_block(dec, z, block_ids):
    """z [B,N,1024] -> {block_id: cls token [B,1152]} for the requested blocks."""
    x = dec.decoder_embed(z)
    x = dec.interpolate_latent(x)
    cls = dec.trainable_cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = x + dec.decoder_pos_embed
    want = set(block_ids)
    out = {}
    for j, layer in enumerate(dec.decoder_layers):
        x = layer(x, head_mask=None)[0]
        if j in want:
            out[j] = x[:, 0, :]
    return out


@torch.no_grad()
def extract_shard(args, dev, lo, hi):
    """Extract cached probe features for image range [lo, hi). One GPU per shard."""
    arr = np.load(args.val_npz, mmap_mode="r")["arr_0"]
    enc = create_encoder("dinov3mls-vit-l16[layers=" + ".".join(map(str, K23_LAYERS)) + "]",
                         device=dev, resolution=256).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    mean, std = MEAN.to(dev), STD.to(dev)
    k23_idx = {l: i for i, l in enumerate(K23_LAYERS)}

    # param-free combines (eval -> mean over fed layers + L23 cls surrogate, no drop)
    cb_k23 = MLSCombine(layers=K23_LAYERS, weighting="mean", cls_surrogate=True,
                        projector="none", dim=1024, out_dim=1024).to(dev).eval()
    cb_k7 = MLSCombine(layers=K7_LAYERS, weighting="mean", cls_surrogate=True,
                       projector="none", dim=1024, out_dim=1024).to(dev).eval()

    decs = {
        "ours": load_decoder_from_ckpt(args.ours_ckpt, dev),
        "k7": load_decoder_from_ckpt(args.k7_decoder, dev),
        "k23": load_decoder_from_ckpt(args.k23_decoder, dev),
    }
    n_blocks = len(decs["ours"].decoder_layers)
    block_ids = list(range(0, n_blocks, DEC_BLOCK_STRIDE))
    feeds = {"ours": "k23", "k7": "k7", "k23": "k23"}      # which latent each decoder eats
    n = hi - lo
    print(f"[shard {args.shard}] range [{lo},{hi}) n={n}; blocks={block_ids} (of {n_blocks})", flush=True)

    Fe = {l: torch.empty(n, 1024, dtype=torch.float16) for l in ENC_PROBE_LAYERS}
    Fd = {m: {b: torch.empty(n, 1152, dtype=torch.float16) for b in block_ids} for m in decs}
    for i in range(lo, hi, args.batch):
        j = min(i + args.batch, hi)
        imgs = torch.stack([torch.from_numpy(arr[k].copy()) for k in range(i, j)])
        imgs = imgs.permute(0, 3, 1, 2).float().to(dev) / 255
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inter = list(enc.model.get_intermediate_layers(
                (imgs - mean) / std, n=K23_LAYERS, reshape=False,
                return_class_token=True, norm=True))
            patch_all = [o[0] for o in inter]
            for l in ENC_PROBE_LAYERS:
                Fe[l][i - lo:j - lo] = inter[k23_idx[l]][1].half().cpu()
            z = {
                "k23": cb_k23(patch_all, idx=None),
                "k7": cb_k7([patch_all[k23_idx[l]] for l in K7_LAYERS], idx=None),
            }
            for m, dec in decs.items():
                blk = decoder_cls_per_block(dec, z[feeds[m]], block_ids)
                for b in block_ids:
                    Fd[m][b][i - lo:j - lo] = blk[b].half().cpu()
        if (j - lo) % 2000 == 0:
            print(f"  [shard {args.shard}] {j - lo}/{n}", flush=True)
    return {"lo": lo, "hi": hi, "block_ids": block_ids, "n_blocks": n_blocks, "Fe": Fe, "Fd": Fd}


def probe(X, y, dev, epochs=30):
    """standardize + Linear(d,1000); train first-40/class, eval last-10/class."""
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
    return acc, int(va.sum())


def run_extract(args, dev):
    arr_n = len(np.load(args.val_npz, mmap_mode="r")["arr_0"])
    chunk = (arr_n + args.nshards - 1) // args.nshards
    lo, hi = args.shard * chunk, min((args.shard + 1) * chunk, arr_n)
    shard = extract_shard(args, dev, lo, hi)
    path = os.path.join(args.out_dir, f"feat_shard{args.shard}.pt")
    torch.save(shard, path)
    print(f"[shard {args.shard}] saved -> {path}", flush=True)


def run_probe(args, dev):
    shards = [torch.load(os.path.join(args.out_dir, f"feat_shard{s}.pt"), map_location="cpu")
              for s in range(args.nshards)]
    shards.sort(key=lambda d: d["lo"])
    block_ids = shards[0]["block_ids"]
    n_blocks = shards[0]["n_blocks"]
    N = shards[-1]["hi"]
    y = torch.arange(N) // 50
    cat_e = lambda l: torch.cat([s["Fe"][l] for s in shards], 0)
    cat_d = lambda m, b: torch.cat([s["Fd"][m][b] for s in shards], 0)

    print("training per-layer probes...", flush=True)
    accs = {}
    val_size = None
    for l in ENC_PROBE_LAYERS:
        a, val_size = probe(cat_e(l), y, dev, args.epochs)
        accs[f"enc_L{l}"] = a
    print(f"  encoder done: { {l: round(accs[f'enc_L{l}'],1) for l in ENC_PROBE_LAYERS} }", flush=True)
    for m in ("ours", "k7", "k23"):
        for b in block_ids:
            a, _ = probe(cat_d(m, b), y, dev, args.epochs)
            accs[f"{m}_b{b}"] = a
        print(f"  {m} done: { {b: round(accs[f'{m}_b{b}'],1) for b in block_ids} }", flush=True)

    out = {
        "enc_probe_layers": ENC_PROBE_LAYERS,
        "block_ids": block_ids,
        "n_blocks": n_blocks,
        "enc_keys": [f"enc_L{l}" for l in ENC_PROBE_LAYERS],
        "ours_keys": [f"ours_b{b}" for b in block_ids],
        "k7_keys": [f"k7_b{b}" for b in block_ids],
        "k23_keys": [f"k23_b{b}" for b in block_ids],
        "val_size": val_size,
        "history": [{"epoch": 1, **accs}],
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {os.path.join(args.out_dir, 'results.json')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["extract", "probe"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--val-npz", default="data_eval/imagenet-256-val.npz")
    ap.add_argument("--ours-ckpt", default="output_full/gdrive_omnirae/omnirae_ckpt.pt")
    ap.add_argument("--k7-decoder", default="pretrained_models/stage1/imagenet/dinov3l-k7/decoder.pt")
    ap.add_argument("--k23-decoder", default="pretrained_models/stage1/imagenet/dinov3l-k23/decoder.pt")
    ap.add_argument("--out-dir", default="output_full/semantic_probe")
    args = ap.parse_args()
    dev = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode == "extract":
        run_extract(args, dev)
    else:
        run_probe(args, dev)


if __name__ == "__main__":
    main()
