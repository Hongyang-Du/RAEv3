#!/usr/bin/env python3
"""Val-PSNR-vs-step curves for the 3 decoder experiments, from each run's
<out_dir>/val_psnr_steps.tsv (written every val_every_steps by train_decoder.py).
Fixed 1000-image ImageNet-val subset (seed 0) -> curves are directly comparable.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {
    "Random Drop Layer MLS + MLP + SIGReg": ("output_full/decoder_random_drop_layer_mls_mlp_sigreg_k23", "tab:red"),
    "Random Drop Layer MLS (-> decoder)":   ("output_full/decoder_random_drop_layer_mls_plain_k23",      "tab:blue"),
    "raev2 K=23":                            ("output_full/decoder_raev2_k23",                            "tab:gray"),
}


def read_tsv(path):
    steps, psnr = [], []
    if not os.path.exists(path):
        return steps, psnr
    for ln in open(path):
        if ln.startswith("step"):
            continue
        s, p = ln.split("\t")[:2]
        steps.append(int(s)); psnr.append(float(p))
    return steps, psnr


fig, ax = plt.subplots(figsize=(9, 5.5))
for name, (d, c) in RUNS.items():
    steps, psnr = read_tsv(os.path.join(d, "val_psnr_steps.tsv"))
    if not steps:
        print(f"(no data yet: {d})")
        continue
    ax.plot(steps, psnr, "o-", color=c, ms=3, lw=1.6,
            label=f"{name} (last {psnr[-1]:.2f} dB)")

ax.set_xlabel("optimizer step")
ax.set_ylabel("val PSNR [dB]  (1000 random ImageNet-val images, seed 0)")
ax.set_title("Stage-1 decoder reconstruction PSNR vs training step", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
out = "output_full/val_psnr_steps_compare.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved -> {out}")
