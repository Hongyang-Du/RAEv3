#!/usr/bin/env python3
"""eval_recon_subset.py + rFID, mask-conditioning aware.

Same architecture/mask handling as src/eval_recon_subset.py (EMA weights, auto-detected
mask-cond decoder with matched k-hot layer_mask, always ImageNet de-norm -> the oldnorm
5ep-sweep convention), PLUS rFID(tf) and, with FD_EVAL=1, the official rFID(fd). Lets the
anchor / maskcond (MLSCombine) and depthattn (DepthAttnCombine) 5ep ckpts be scored on the
full 50k official val with the SAME feed defs as run_variant_sweep_5ep_eval.sh
(k23full / k7=idx 0..6 / l11=idx 10), so the 50k rFID extends that 1000-img PSNR/SSIM table.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from omegaconf import OmegaConf
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

from encoders.vision_encoder import create_encoder
from stage1.rae import _load_decoder
from utils.model_utils import get_obj_from_str
from eval.fid import calculate_rfid

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def per_image_psnr(rec, ref):
    mse = ((rec.clamp(0, 1) - ref) ** 2).mean(dim=(1, 2, 3))
    return -10 * torch.log10(mse.clamp_min(1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--idx", default=None, help="comma-sep 0-based positions into layers")
    ap.add_argument("--num-images", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val-npz", default=None, help="override eval.val_npz")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--null-cond", action="store_true",
                    help="mask-cond ckpts: decode with the learned null embedding")
    ap.add_argument("--save-recon-npz", default=None)
    args = ap.parse_args()
    device = torch.device("cuda")

    cfg = OmegaConf.load(args.config)
    ev = cfg.get("eval", {}) or {}
    ckpt_path = args.ckpt or ev.get("ckpt")
    if not ckpt_path:
        ap.error("no ckpt: pass --ckpt or set eval.ckpt in the config")
    num_images = args.num_images or ev.get("num_images", 1000)
    val_npz = args.val_npz or ev.get("val_npz", "data_eval/imagenet-256-val.npz")
    seed = args.seed if args.seed is not None else ev.get("seed", 0)
    batch = args.batch
    tag = args.tag or ev.get("tag", "")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    layers = list(ck["layers"])

    if args.idx:
        idx = [int(x) for x in args.idx.split(",")]
    elif ev.get("subset_layers"):
        subset = list(ev["subset_layers"])
        missing = [L for L in subset if L not in layers]
        if missing:
            raise ValueError(f"subset_layers {missing} not in ckpt layers {layers}")
        idx = [layers.index(L) for L in subset]
    else:
        idx = None
    used = [layers[i] for i in idx] if idx is not None else layers
    print(f"[{tag}] ckpt={os.path.basename(ckpt_path)} ep={ck.get('epoch')}\n"
          f"  trained layers={layers}\n  eval layers (idx={idx}) -> {used}", flush=True)

    enc = create_encoder("dinov3mls-vit-l16[layers=" + ".".join(map(str, layers)) + "]",
                         device=device, resolution=256).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    mean, std = MEAN.to(device), STD.to(device)

    params = OmegaConf.to_container(cfg.combine.params, resolve=True)
    combine = get_obj_from_str(cfg.combine.target)(**params).to(device).eval()
    combine.load_state_dict(ck["ema_combine"])
    mc_emb = ck["ema_dec"].get("mask_embedder.layer_emb")
    mask_cond = {"K": mc_emb.shape[0], "d_c": mc_emb.shape[1]} if mc_emb is not None else None
    if mask_cond:
        print(f"  mask_cond detected (K={mask_cond['K']}, d_c={mask_cond['d_c']})"
              + ("  [NULL-cond A/B]" if args.null_cond else "  [matched-mask cond]"), flush=True)
    dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                        num_patches=256, pretrained_path=None,
                        mask_cond=mask_cond).to(device).eval()
    dec.load_state_dict(ck["ema_dec"])
    del ck

    arr = np.load(val_npz, mmap_mode="r")
    arr = arr[arr.files[0]] if hasattr(arr, "files") else arr
    g = torch.Generator().manual_seed(seed)
    idxs = torch.randperm(len(arr), generator=g)[:num_images].tolist()

    psnrs, ssims, n_done = [], [], 0
    recon_u8, ref_u8 = [], []
    with torch.no_grad():
        for i in range(0, len(idxs), batch):
            bi = idxs[i:i + batch]
            imgs = torch.stack([torch.from_numpy(arr[j].copy()) for j in bi])
            imgs = imgs.permute(0, 3, 1, 2).float().to(device) / 255
            with torch.autocast("cuda", dtype=torch.bfloat16):
                toks = list(enc.model.get_intermediate_layers(
                    (imgs - mean) / std, n=layers, reshape=False,
                    return_class_token=False, norm=True))
                z = combine(toks, idx=idx)
                lm = None
                if mask_cond and not args.null_cond:
                    lm = torch.zeros(imgs.shape[0], len(layers), dtype=torch.bool, device=device)
                    lm[:, idx if idx is not None else slice(None)] = True
                out = dec(z, drop_cls_token=False, layer_mask=lm).logits
                rec = dec.unpatchify(out) * std + mean
            rec = rec.clamp(0, 1).float()
            psnrs.append(per_image_psnr(rec, imgs))
            ssims.append(ssim_fn(rec, imgs, data_range=1.0).item() * rec.shape[0])
            recon_u8.append(rec.mul(255).round().permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy())
            ref_u8.append(imgs.mul(255).round().permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy())
            n_done += imgs.shape[0]
            if n_done % 3200 == 0:
                print(f"  {n_done}/{len(idxs)}", flush=True)

    psnr = torch.cat(psnrs)
    ssim = sum(ssims) / n_done
    recon_arr = np.concatenate(recon_u8, 0)
    ref_arr = np.concatenate(ref_u8, 0)

    if args.save_recon_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_recon_npz)), exist_ok=True)
        np.savez(args.save_recon_npz, recon=recon_arr, idxs=np.asarray(idxs, dtype=np.int64))
        print(f"[save] recon npz -> {args.save_recon_npz}", flush=True)

    # NB: calculate_rfid tests `cuda=(device=="cuda")` -> must pass the STRING "cuda",
    # NOT torch.device("cuda") (which != "cuda" -> silently runs inception on CPU, ~15x slower).
    rfid = calculate_rfid(recon_arr, ref_arr, bs=128, device="cuda")
    fd_rfid = None
    if os.environ.get("FD_EVAL"):
        from eval.distributional import compute_distributional_metrics
        fd_out = compute_distributional_metrics(
            recon_arr, ["fid"], reference_npz=None, data_dir=None,
            device=device, batch_size=128)
        fd_rfid = float(fd_out["fid"])

    print(f"\n[{tag}] N={n_done}  eval_layers={used}")
    line = (f"  PSNR = {psnr.mean():.3f} dB   SSIM = {ssim:.4f}   rFID(tf) = {rfid:.3f}")
    if fd_rfid is not None:
        line += f"   rFID(fd:{os.environ.get('FID_REFERENCE','imagenet_256')}) = {fd_rfid:.3f}"
    print(line)

    out = args.out or ckpt_path.replace(".pt", f"_reconrfid{('_'+tag) if tag else ''}.json")
    with open(out, "w") as f:
        json.dump({"ckpt": ckpt_path, "tag": tag, "eval_layers": used, "num_images": n_done,
                   "mask_cond": bool(mask_cond), "null_cond": bool(args.null_cond),
                   "psnr": psnr.mean().item(), "ssim": ssim,
                   "rfid_tf": float(rfid), "rfid_fd": fd_rfid,
                   "fd_reference": os.environ.get("FID_REFERENCE", "imagenet_256") if fd_rfid is not None else None},
                  f, indent=2)
    print(f"json -> {out}")


if __name__ == "__main__":
    main()
