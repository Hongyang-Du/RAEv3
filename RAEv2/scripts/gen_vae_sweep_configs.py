#!/usr/bin/env python3
"""Generate the out_dim compression-rate sweep configs (5-epoch smoke sweep, Variant B
depth-attn fusion + VAE bottleneck) from the out_dim=4 template
(randomdrop-plain-k23-nano-p03-depthattn-vae4-5ep-oldnorm.yaml).

Everything except out_dim/decoder.latent_dim/out_dir/wandb.name is held fixed across
the sweep (same inner DepthAttnCombine at dim=out_dim=1024, same p_drop=0.3, same
5-epoch/disc_start=2 budget) -- out_dim is the only swept axis.

Usage:
    python scripts/gen_vae_sweep_configs.py
"""
import os

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                          "configs", "stage1", "decoder")
TEMPLATE = os.path.join(CONFIG_DIR, "randomdrop-plain-k23-nano-p03-depthattn-vae4-5ep-oldnorm.yaml")
SWEEP_OUT_DIMS = [16, 32, 64, 128, 256, 512]     # 4 already exists (the template itself)


def main():
    with open(TEMPLATE) as f:
        template = f.read()

    for out_dim in SWEEP_OUT_DIMS:
        text = template
        text = text.replace("VAEBottleneck(out_dim=4)", f"VAEBottleneck(out_dim={out_dim})")
        text = text.replace("(decoder.latent_dim=4)", f"(decoder.latent_dim={out_dim})")
        text = text.replace("\n    out_dim: 4\n    variational: true",
                            f"\n    out_dim: {out_dim}\n    variational: true")
        text = text.replace("\n  latent_dim: 4\n", f"\n  latent_dim: {out_dim}\n")
        text = text.replace("depthattn-vae4-5ep-sweep", f"depthattn-vae{out_dim}-5ep-sweep")

        assert text != template, f"no substitution happened for out_dim={out_dim} -- template pattern drifted"

        out_path = os.path.join(CONFIG_DIR,
                                f"randomdrop-plain-k23-nano-p03-depthattn-vae{out_dim}-5ep-oldnorm.yaml")
        with open(out_path, "w") as f:
            f.write(text)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
