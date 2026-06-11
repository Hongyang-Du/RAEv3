#!/usr/bin/env python3
"""Probe per-layer usage of a trained MLS projector+decoder ckpt (gate-free collapse readout).

The nogate / dropmean combine has NO parameters, so the same projector+decoder can
decode from ANY layer subset. Two profiles replace reading gate weights:

  - LOO dPSNR_i  = full PSNR - PSNR without layer i   -> reliance on layer i
  - solo PSNR_i  = PSNR from layer i alone            -> sufficiency of layer i

Collapse signature: one layer (L11) dominates both — huge LOO drop, solo ~= full,
while deep layers show ~0 LOO drop and poor solo. Healthy fusion: LOO spread thin
across layers, all solos clearly below full.

Works on nogate and dropmean ckpts (identical projector params). NOT for the softgate
ckpt (its projector was trained on gate-weighted input, subsets are OOD for it).

Usage (inside rae container, single GPU, no DDP):
    python src/probe_layer_usage.py \
        --ckpt output_full/train_decoder_mls_nogate_sigreg/ckpt_latest.pt \
        --val-image assets/samples/sample_1.png
"""
import sys, os, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torchvision import transforms

from train_decoder_mls_dropmean_sigreg import MLSProjector
from overfit_sigreg import psnr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       required=True)
    ap.add_argument("--val-image",  default="assets/samples/sample_1.png",
                    help="Fixed image (its dir is globbed, same as training)")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--latent-dim", type=int, default=1024)
    ap.add_argument("--device",     default="cuda:0")
    ap.add_argument("--raw",        action="store_true", help="Use raw weights instead of EMA")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    layers = list(ckpt.get("layers", [11, 13, 15, 17, 19, 21, 23]))
    print(f"ckpt: {args.ckpt}  epoch={ckpt.get('epoch')}  layers={layers}"
          f"  weights={'raw' if args.raw else 'EMA'}")

    # -- encoder (frozen) --------------------------------------------------------
    from encoders.vision_encoder import create_encoder
    layers_str = ".".join(str(l) for l in layers)
    encoder = create_encoder(f"dinov3mls-vit-l16[layers={layers_str}]",
                             device=device, resolution=args.image_size)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    enc_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    enc_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # -- val images (same globbing as training) -----------------------------------
    from PIL import Image as PILImage
    t = transforms.Compose([
        transforms.Resize(args.image_size + 32),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])
    val_dir = os.path.dirname(os.path.abspath(args.val_image))
    paths = sorted(
        p for p in (glob.glob(os.path.join(val_dir, "*.png")) +
                    glob.glob(os.path.join(val_dir, "*.jpg")))
        if "concat" not in os.path.basename(p).lower()
    ) or [args.val_image]
    imgs = torch.stack([t(PILImage.open(p).convert("RGB")) for p in paths]).to(device)
    with torch.no_grad():
        toks = list(encoder.model.get_intermediate_layers(
            (imgs - enc_mean) / enc_std, n=encoder.layer_indices, reshape=False,
            return_class_token=False, norm=True))

    # -- projector + decoder from ckpt --------------------------------------------
    projector = MLSProjector(dim=encoder.hidden_size, out_dim=args.latent_dim).to(device).eval()
    projector.load_state_dict(ckpt["projector" if args.raw else "ema_proj"])

    from omegaconf import OmegaConf
    from stage1.rae import _load_decoder
    s1 = OmegaConf.load("configs/stage2/training/imagenet-dinov3l-k7.yaml").stage_1.params
    decoder = _load_decoder(
        s1.decoder_config_path,
        hidden_size=args.latent_dim, patch_size=16, num_patches=256,
        pretrained_path=None,
    ).to(device).eval()
    decoder.load_state_dict(ckpt["decoder" if args.raw else "ema_dec"])
    for p in list(projector.parameters()) + list(decoder.parameters()):
        p.requires_grad_(False)

    @torch.no_grad()
    def subset_psnr(idx):
        z = projector([toks[i] for i in idx])           # eval mode -> plain mean over list
        d = decoder(z, drop_cls_token=False).logits
        rec = (decoder.unpatchify(d) * enc_std + enc_mean).clamp(0, 1)
        return psnr(rec, imgs)

    K = len(layers)
    full = subset_psnr(list(range(K)))
    print(f"\n{len(paths)} val images   full-mean PSNR = {full:.2f} dB\n")
    print(f"{'layer':>6} | {'solo PSNR':>9} | {'LOO PSNR':>8} | {'LOO dPSNR (reliance)':>20}")
    print("-" * 56)
    for i, L in enumerate(layers):
        solo = subset_psnr([i])
        loo  = subset_psnr([j for j in range(K) if j != i])
        print(f"L{L:>5} | {solo:9.2f} | {loo:8.2f} | {full - loo:+20.2f}")
    no_shallow = subset_psnr(list(range(1, K)))
    deep3      = subset_psnr(list(range(K - 3, K)))
    print(f"\nwithout L{layers[0]} (deep-only): {no_shallow:.2f} dB"
          f"   deepest-3 {layers[-3:]}: {deep3:.2f} dB")


if __name__ == "__main__":
    main()
