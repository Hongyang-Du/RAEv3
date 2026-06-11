#!/usr/bin/env python3
"""Plot Val PSNR (EMA) vs epoch from training stdout logs — a wandb-free comparison.

Each training prints one `Val PSNR (EMA): X dB` line per epoch, so the Nth such line
is epoch N. Usage:

    python plot_val_psnr.py                       # default: the 3 output_full runs
    python plot_val_psnr.py "label=path/train.log" ...   # custom runs
    python plot_val_psnr.py --out foo.png "..."   # custom output image
"""
import re, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RUNS = [
    "raev2 (MLS)=output_full/train_decoder_mls_raev2/train.log",
    "SIGReg(nogate)=output_full/train_decoder_mls_nogate_sigreg/train.log",
    "SIGReg(softgate)=output_full/train_decoder_mls_softgate_sigreg/train.log",
    "SIGReg(dropmean)=output_full/train_decoder_mls_dropmean_sigreg/train.log",
]

args = sys.argv[1:]
out_path = "output_full/val_psnr_compare.png"
if "--out" in args:
    i = args.index("--out")
    out_path = args[i + 1]
    del args[i:i + 2]
runs = args if args else DEFAULT_RUNS

pat = re.compile(r"Val PSNR \(EMA\):\s*([0-9.]+)\s*dB")
plt.figure(figsize=(8, 5))
any_data = False
for spec in runs:
    label, _, path = spec.partition("=")
    if not os.path.exists(path):
        print(f"skip  {label:14s}: {path} (not found)")
        continue
    with open(path, errors="ignore") as f:
        vals = [float(m.group(1)) for m in pat.finditer(f.read())]
    if not vals:
        print(f"skip  {label:14s}: no Val PSNR lines yet ({path})")
        continue
    epochs = list(range(1, len(vals) + 1))
    plt.plot(epochs, vals, marker="o", ms=3, label=f"{label} (best {max(vals):.2f} dB)")
    any_data = True
    print(f"plot  {label:14s}: {len(vals)} epochs, last={vals[-1]:.2f}, best={max(vals):.2f} dB")

if not any_data:
    print("No data to plot yet (no train.log with Val PSNR lines).")
    sys.exit(0)

plt.xlabel("epoch")
plt.ylabel("Val PSNR (EMA)  [dB]")
plt.title("Val PSNR — decoder reconstruction quality")
plt.legend()
plt.grid(alpha=0.3)
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
plt.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"saved -> {out_path}")
