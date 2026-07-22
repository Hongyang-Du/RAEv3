"""Compile the 5-epoch variant-sweep recon eval JSONs (anchor vs Variant A maskcond vs
Variant B depthattn, k23full/k7/l11 feeds) into one CSV + printed comparison table.
Reads <ckpt>_recon_<name>_<feed>.json next to each ckpt_latest.pt (written by
run_variant_sweep_5ep_eval.sh -> src/eval_recon_subset.py)."""
import json
import os
import csv

ROOT = "/sensei-fs-3/users/hongyangd/ckpt"
OUTDIR = {
    "anchor": "omni-randomdrop-plain-k23-nano-p0.3-5ep-sweep",
    "maskcond": "omni-randomdrop-plain-k23-nano-p0.3-maskcond-5ep-sweep",
    "depthattn": "omni-randomdrop-plain-k23-nano-p0.3-depthattn-5ep-sweep",
}
FEEDS = ["k23full", "k7", "l11"]
OUT = "/sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2/variant_sweep_5ep_recon.csv"

rows = []
for variant, d in OUTDIR.items():
    tags = list(FEEDS) + (["k23full_null"] if variant == "maskcond" else [])
    for feed in tags:
        fn = f"ckpt_latest_recon_{variant}_{feed}.json"
        p = os.path.join(ROOT, d, fn)
        if not os.path.isfile(p):
            rows.append({"variant": variant, "feed": feed, "num_images": "MISSING",
                         "psnr": "", "ssim": "", "eval_layers": ""})
            continue
        j = json.load(open(p))
        rows.append({
            "variant": variant, "feed": feed, "num_images": j.get("num_images"),
            "psnr": round(j["psnr_mean"], 3), "ssim": round(j["ssim"], 4),
            "eval_layers": "|".join(map(str, j.get("eval_layers", []))),
        })

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["variant", "feed", "num_images", "psnr", "ssim", "eval_layers"])
    w.writeheader()
    w.writerows(rows)

print(f"-> {OUT}\n")
print(f"{'variant':<12}{'feed':<16}{'N':>7}{'PSNR':>9}{'SSIM':>9}")
for r in rows:
    print(f"{r['variant']:<12}{r['feed']:<16}{str(r['num_images']):>7}{str(r['psnr']):>9}{str(r['ssim']):>9}")
