#!/usr/bin/env python3
"""Aggregate gen_crossdecode shards and report the comparison table:
  distribution (vs real ImageNet val):  gFID, IS, KID   for native-k7 and ours
  decoder agreement (ours vs native, same latent):  PSNR, SSIM, LPIPS

  python eval_crossdecode.py --dir output_full/crossdec_k7 --nshards 8
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
from torch_fidelity import calculate_metrics
from eval.utils import ImgArrDataset


def load_set(d, prefix, nshards):
    return np.concatenate([np.load(f"{d}/{prefix}_{i}.npy") for i in range(nshards)])


def dist_metrics(gen, real, bs=128):
    m = calculate_metrics(input1=ImgArrDataset(gen), input2=ImgArrDataset(real),
                          fid=True, isc=True, kid=True,
                          kid_subset_size=min(1000, len(gen)), batch_size=bs, cuda=True, verbose=False)
    return (m["frechet_inception_distance"], m["inception_score_mean"],
            m["kernel_inception_distance_mean"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output_full/crossdec_k7")
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--real", default="data_eval/imagenet-256-val.npz")
    args = ap.parse_args()

    native = load_set(args.dir, "native", args.nshards)
    ours = load_set(args.dir, "ours", args.nshards)
    real = np.load(args.real, mmap_mode="r")["arr_0"]
    N = len(native)
    print(f"gen N={N}  real N={len(real)}", flush=True)

    ag = [np.load(f) for f in sorted(glob.glob(f"{args.dir}/agree_*.npz"))]
    n = sum(float(a["n"]) for a in ag)
    psnr = sum(float(a["psnr"]) for a in ag) / n
    ssim = sum(float(a["ssim"]) for a in ag) / n
    lpip = sum(float(a["lpips"]) for a in ag) / n

    fid_n, is_n, kid_n = dist_metrics(native, real)
    print(f"native done", flush=True)
    fid_o, is_o, kid_o = dist_metrics(ours, real)
    print(f"ours done", flush=True)

    print("\n================ k7-DiT cross-decode (N=%d, vs ImageNet val) ================" % N)
    print(f"{'decoder':<18}{'gFID↓':>9}{'IS↑':>9}{'KID↓':>11}")
    print(f"{'RAEv2-k7 (native)':<18}{fid_n:>9.3f}{is_n:>9.3f}{kid_n:>11.5f}")
    print(f"{'OmniRAE (ours)':<18}{fid_o:>9.3f}{is_o:>9.3f}{kid_o:>11.5f}")
    print(f"\nDecoder agreement (ours vs native, same latent, N={int(n)}):")
    print(f"  PSNR={psnr:.2f} dB   SSIM={ssim:.4f}   LPIPS={lpip:.4f}")

    import json
    json.dump(dict(N=N, native=dict(gfid=fid_n, is_=is_n, kid=kid_n),
                   ours=dict(gfid=fid_o, is_=is_o, kid=kid_o),
                   agreement=dict(psnr=psnr, ssim=ssim, lpips=lpip)),
              open(f"{args.dir}/results.json", "w"), indent=2)
    print(f"\njson -> {args.dir}/results.json")


if __name__ == "__main__":
    main()
