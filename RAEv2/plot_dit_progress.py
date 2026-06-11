#!/usr/bin/env python3
"""Plot DiT denoise-probe PSNR vs epoch across the 3 stage-2 runs (wandb-free compare).

Each stage-2 epoch logs one line:
  [Epoch N] Denoise PSNR (EMA): t25=..  t50=..  t75=..  t95=..  ceil=.. dB
Pixel-space, same probe images/noise across runs -> directly comparable.

    python plot_dit_progress.py                      # default: 3 runs below
    python plot_dit_progress.py "label=path/train.log" ...
"""
import re, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RUNS = [
    "raev2 (MLS)=ckpts_full/stage2/dit-raev2mls-k7/train.log",
    "SIGReg(nogate)=ckpts_full/stage2/dit-nogate-k7/train.log",
    "SIGReg(dropmean)=ckpts_full/stage2/dit-dropmean-k7/train.log",
]

args = sys.argv[1:]
out_path = "ckpts_full/stage2/dit_denoise_compare.png"
if "--out" in args:
    i = args.index("--out")
    out_path = args[i + 1]
    del args[i:i + 2]
runs = args if args else DEFAULT_RUNS

line_pat = re.compile(r"Denoise PSNR \(EMA\):\s*(.*?)\s*ceil=([0-9.]+)")
t_pat = re.compile(r"t(\d+)=([0-9.]+)")

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
t_keys = [25, 50, 75, 95]
ax_by_t = dict(zip(t_keys, axes.flat))
colors = plt.cm.tab10.colors
any_data = False

for ri, spec in enumerate(runs):
    label, _, path = spec.partition("=")
    if not os.path.exists(path):
        print(f"skip  {label:16s}: {path} (not found)")
        continue
    per_t = {k: [] for k in t_keys}
    ceil = []
    with open(path, errors="ignore") as f:
        for m in line_pat.finditer(f.read()):
            vals = dict((int(k), float(v)) for k, v in t_pat.findall(m.group(1)))
            for k in t_keys:
                if k in vals:
                    per_t[k].append(vals[k])
            ceil.append(float(m.group(2)))
    if not ceil:
        print(f"skip  {label:16s}: no Denoise PSNR lines yet ({path})")
        continue
    epochs = list(range(1, len(ceil) + 1))
    for k in t_keys:
        ax = ax_by_t[k]
        ax.plot(epochs, per_t[k], marker="o", ms=2.5, color=colors[ri % 10],
                label=f"{label} (last {per_t[k][-1]:.2f})")
        ax.axhline(ceil[-1], color=colors[ri % 10], ls="--", lw=0.8, alpha=0.5)
    any_data = True
    print(f"plot  {label:16s}: {len(ceil)} epochs, ceil={ceil[-1]:.2f} dB, "
          + " ".join(f"t{k}={per_t[k][-1]:.2f}" for k in t_keys))

if not any_data:
    print("No data to plot yet.")
    sys.exit(0)

for k in t_keys:
    ax = ax_by_t[k]
    ax.set_title(f"t = 0.{k:02d}" if k < 100 else f"t={k/100}")
    ax.set_ylabel("Denoise PSNR (EMA) [dB]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("epoch")
fig.suptitle("DiT denoise probe — x-prediction PSNR in pixel space (dashed = stage-1 ceiling)")
fig.tight_layout()
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
fig.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"saved -> {out_path}")
