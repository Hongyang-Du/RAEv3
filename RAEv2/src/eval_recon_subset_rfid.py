"""eval_recon_subset.py + rFID. Reconstruction PSNR / SSIM / rFID for a MLSCombine
stage-1 ckpt with an optional layer-subset feed (one trained model, fewer fed layers).

  python src/eval_recon_subset_rfid.py --config configs/stage1/decoder/infer-k23plain-fed-k7.yaml \
      --idx 0,1,..,22 --num-images 10000 --tag plain_k23_10k
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
    ap.add_argument("--val-npz", default=None, help="override eval.val_npz (e.g. a held-out eval set)")
    ap.add_argument("--seed", type=int, default=None, help="override eval.seed (subset sampling)")
    args = ap.parse_args()
    device = "cuda"

    cfg = OmegaConf.load(args.config)
    ev = cfg.get("eval", {}) or {}
    tag = args.tag or ev.get("tag", "")
    ckpt_path = args.ckpt or ev.get("ckpt")
    if not ckpt_path:
        ap.error("no ckpt: pass --ckpt or set eval.ckpt in the config")
    val_npz = args.val_npz or ev.get("val_npz", "data_eval/imagenet-256-val.npz")
    num_images = args.num_images or ev.get("num_images", 1000)
    seed = args.seed if args.seed is not None else ev.get("seed", 0)
    batch = args.batch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    layers = list(cfg.combine.params.layers)
    if args.idx:
        idx = [int(x) for x in args.idx.split(",")]
    else:
        subset = ev.get("subset_layers", None)
        if subset:
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

    # USE_EMA=0 -> load raw (non-EMA) combine/decoder weights instead of the EMA copies.
    # (ckpt key naming is asymmetric: EMA=ema_combine/ema_dec, non-EMA=combine/decoder)
    _ema = os.environ.get("USE_EMA", "1") != "0"
    comb_key = "ema_combine" if _ema else "combine"
    dec_key = "ema_dec" if _ema else "decoder"
    print(f"[weights] {'EMA' if _ema else 'non-EMA'}: combine[{comb_key}] dec[{dec_key}]", flush=True)
    params = OmegaConf.to_container(cfg.combine.params, resolve=True)
    combine = get_obj_from_str(cfg.combine.target)(**params).to(device).eval()
    combine.load_state_dict(ck[comb_key])
    dec = _load_decoder("configs/decoder/ViTXL", hidden_size=1024, patch_size=16,
                        num_patches=256, pretrained_path=None).to(device).eval()
    dec.load_state_dict(ck[dec_key])
    del ck

    arr = np.load(val_npz, mmap_mode="r")
    arr = arr[arr.files[0]] if hasattr(arr, "files") else arr
    g = torch.Generator().manual_seed(seed)
    idxs = torch.randperm(len(arr), generator=g)[:num_images].tolist()

    psnrs, ssims, n_done = [], [], 0
    recon_u8, ref_u8 = [], []                       # for rFID
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
                out = dec(z, drop_cls_token=False).logits
                # NO_DENORM=1 for decoders trained to output [0,1] DIRECTLY (new RAE-style
                # convention); default applies ImageNet de-norm for the old normalized-output decoders.
                rec = dec.unpatchify(out) if os.environ.get("NO_DENORM") else dec.unpatchify(out) * std + mean
            rec = rec.clamp(0, 1).float()
            psnrs.append(per_image_psnr(rec, imgs))
            ssims.append(ssim_fn(rec, imgs, data_range=1.0).item() * rec.shape[0])
            recon_u8.append(rec.mul(255).round().permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy())
            ref_u8.append(imgs.mul(255).round().permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy())
            n_done += imgs.shape[0]
            if n_done % 320 == 0:
                print(f"  {n_done}/{len(idxs)}", flush=True)

    psnr = torch.cat(psnrs)
    ssim = sum(ssims) / n_done
    recon_arr = np.concatenate(recon_u8, 0)
    ref_arr = np.concatenate(ref_u8, 0)
    rfid = calculate_rfid(recon_arr, ref_arr, bs=128, device=device)

    # FD_EVAL=1: also compute the OFFICIAL rFID via fd_evaluator on the SAME recon_arr
    # (no re-generation). reference_npz=None routes to fd_evaluator; fid_reference falls
    # back to imagenet_256_fid_stats unless FID_REFERENCE overrides (e.g. guided_diffusion_stats).
    fd_rfid = None
    if os.environ.get("FD_EVAL"):
        from eval.distributional import compute_distributional_metrics
        fd_out = compute_distributional_metrics(
            recon_arr, ["fid"], reference_npz=None, data_dir=None,
            device=torch.device(device), batch_size=128)
        fd_rfid = float(fd_out["fid"])

    print(f"\n[{tag}] N={n_done}  eval_layers={used}")
    line = f"  PSNR = {psnr.mean():.3f} dB   SSIM = {ssim:.4f}   rFID(tf) = {rfid:.3f}"
    if fd_rfid is not None:
        line += f"   rFID(fd:{os.environ.get('FID_REFERENCE','imagenet_256')}) = {fd_rfid:.3f}"
    print(line)

    out = args.out or ckpt_path.replace(".pt", f"_reconrfid{('_'+tag) if tag else ''}.json")
    with open(out, "w") as f:
        json.dump({"ckpt": ckpt_path, "tag": tag, "eval_layers": used, "num_images": n_done,
                   "psnr": psnr.mean().item(), "ssim": ssim,
                   "rfid_tf": float(rfid), "rfid_fd": fd_rfid,
                   "fd_reference": os.environ.get("FID_REFERENCE", "imagenet_256") if fd_rfid is not None else None},
                  f, indent=2)
    print(f"json -> {out}")


if __name__ == "__main__":
    main()
