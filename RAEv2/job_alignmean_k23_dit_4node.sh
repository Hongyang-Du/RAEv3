#!/usr/bin/env bash
set -exo pipefail

# Stage-2 DiT on the ALIGN-TO-(mean+cls) k23 depth-attn latent from the TRAINED decoder
# ckpt/alignmean-cls-k23-depthattn (config: configs/stage2/training/
# imagenet-dinov3l-alignmean-cls-depthattn-k23.yaml). use_ema:true, drop:false (full-feed
# latent) -> viz ON, samples decode with ema_dec. 4 nodes x 8 GPU, 80 epochs.
# PREREQ: run run_alignmean_k23_stats.sh ONCE first (writes latent_stats.pt that the config
# points at) -- cls_surrogate:true + loss.align shift the latent off N(0,I) and the DiT
# needs it to standardize.
export WANDB_KEY=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
export WANDB_ENTITY=uscgvl
export WANDB_PROJECT=omnirae

export NUM_NODES=4                          # MUST equal the job replica count

bash /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2/run_pluto_job_4node.sh alignmean-k23
