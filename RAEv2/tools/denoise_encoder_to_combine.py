#!/usr/bin/env python
"""Convert a Stage-0 denoise-AE ckpt into a `combine`-keyed ckpt that
train_decoder.py's STAGE0_COMBINE path can strict-load into a
stage1.denoise_ae.DenoiseAECombine.

The Stage-0 ckpt stores the encoder E under top-level key `encoder` (keys like
`pos`, `blocks.0...`, `proj...`). DenoiseAECombine holds E as `self.E`, so its
state_dict keys are `E.pos`, `E.blocks.0...`, `E.proj...`. We just reprefix and
wrap under `combine` (the key train_decoder.py reads: `_ck0["combine"]`).

usage: python tools/denoise_encoder_to_combine.py <stage0_ckpt.pt> <out_combine.pt>
"""
import sys
import torch

def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu", weights_only=False)
    enc = ck["encoder"]
    combine_sd = {f"E.{k}": v for k, v in enc.items()}
    out = {"combine": combine_sd,
           "epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
           "layers": ck.get("layers"), "dim": ck.get("dim"), "d": ck.get("d"),
           "cls_surrogate": ck.get("cls_surrogate"),
           "_note": f"reprefixed encoder->E from {src}"}
    torch.save(out, dst)
    print(f"[ok] {src}")
    print(f"     encoder keys: {len(enc)}  -> combine keys: {len(combine_sd)}")
    print(f"     dim={ck.get('dim')} d={ck.get('d')} cls_surrogate={ck.get('cls_surrogate')} "
          f"layers(K)={len(ck.get('layers', []))}")
    print(f"     wrote {dst}")

if __name__ == "__main__":
    main()
