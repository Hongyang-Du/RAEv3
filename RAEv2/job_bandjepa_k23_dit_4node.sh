#!/usr/bin/env bash
# Pluto job: Stage-2 DiT on the FROZEN Stage-0 3-band depth-JEPA k23 latent. 4 nodes x 8 GPU.
# CLS-ON lineage (cls_surrogate:true, attn_kind:softmax) -> encodes the SAME latent that the
# trained Stage-1 decoder ckpt/invert-stage0-bandjepa-k23 (val PSNR ~30) inverts, so gFID can
# be computed later without retraining a decoder. On-the-fly encode from the frozen bandjepa
# combine (stage-0 ckpt ckpt/stage0-bandjepa-depthattn-k23, no decoder in it) -> the DiT only
# needs combine.encode(). Sample-viz is OFF (sample_every huge) and eval:null, so the missing
# decoder is never called during training.
#
# PREREQ: latent_stats.pt MUST exist at
#   /sensei-fs-3/users/hongyangd/ckpt/stage0-bandjepa-depthattn-k23/latent_stats.pt
# (the config's normalization_stat_path). cls_surrogate:true shifts the latent off N(0,I), so
# unlike jepa-k7 this run standardizes. Run run_bandjepa_k23_stats.sh ONCE before launching.
#
# Auto-resumes from the run's ckpt_latest. Set replicas = NUM_NODES.
set -exo pipefail

export WANDB_KEY="${WANDB_KEY:-}"          # paste key for online logging (optional)
export WANDB_ENTITY=uscgvl
export WANDB_PROJECT=raev3-full

export NUM_NODES=4                          # MUST equal the job replica count

bash /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2/run_pluto_job_4node.sh bandjepa-k23
