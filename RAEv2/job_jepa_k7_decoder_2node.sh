#!/usr/bin/env bash
# Pluto job: Stage-1 decoder on the FROZEN Stage-0 JEPA k7 fusion. 2 nodes x 8 GPU.
# The combine is loaded + frozen from the stage-0 ckpt via STAGE0_COMBINE (train_decoder.py).
# Auto-resumes from <out_dir>/ckpt_latest.pt. Set replicas = NUM_NODES.
set -exo pipefail

export WANDB_KEY="${WANDB_KEY:-}"          # paste key from job_randomdrop_p07_2node.sh for online logging
export WANDB_ENTITY=uscgvl
export WANDB_PROJECT=raev3-full

export NUM_NODES=2                          # MUST equal the job replica count
export STAGE0_COMBINE=/sensei-fs-3/users/hongyangd/ckpt/stage0-fusion-jepa-depthattn-k7/ckpt_latest.pt
export OUT_DIR_OVERRIDE=/sensei-fs-3/users/hongyangd/ckpt/stage1-decoder-jepa-depthattn-k7

bash /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2/run_pluto_decoder_4node.sh jepa-k7-decoder
