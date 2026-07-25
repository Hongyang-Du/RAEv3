#!/usr/bin/env bash
# Pluto job: Stage-2 DiT on the FROZEN Stage-0 JEPA k7 latent. 4 nodes x 8 GPU.
# On-the-fly encode from the frozen JEPA combine (stage-0 ckpt, no decoder) -> the DiT
# only needs combine.encode(). Sample-viz is OFF (sample_every huge) and eval is null,
# so the missing decoder is never called. gFID comes later, once the Stage-1 decoder
# (job_jepa_k7_decoder_2node.sh) has produced a ckpt to decode with.
# Auto-resumes from the run's ckpt_latest. Set replicas = NUM_NODES.
set -exo pipefail

export WANDB_KEY="${WANDB_KEY:-}"          # paste key from job_randomdrop_p07_2node.sh for online logging
export WANDB_ENTITY=uscgvl
export WANDB_PROJECT=raev3-full

export NUM_NODES=4                          # MUST equal the job replica count

bash /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2/run_pluto_job_4node.sh jepa-k7
