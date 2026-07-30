#!/usr/bin/env bash
# Pluto job: Stage-1 decoder + TRAINABLE k7 depth-attn fusion, align-to-(mean+cls).
# 2 nodes x 8 GPU, 16 epochs, w_align=1.
#
# Differs from job_jepa_k7_decoder_2node.sh: the fusion is TRAINABLE and trained JOINTLY
# with the decoder (recon + align gradients reach it). So we DELIBERATELY do NOT set
# STAGE0_COMBINE -- setting it would freeze the fusion and train_decoder.py's align guard
# would (correctly) abort. The combine trains from scratch (zero-init depth-attn residual
# == starts at mean+cls == loss_align ~0 at init). sigreg is off; loss_align is the anchor.
# Auto-resumes from <out_dir>/ckpt_latest.pt. Set replicas = NUM_NODES.
set -exo pipefail

export WANDB_KEY="${WANDB_KEY:-}"          # paste key for online logging; empty -> offline
export WANDB_ENTITY=uscgvl
export WANDB_PROJECT=raev3-full
[ -z "${WANDB_KEY}" ] && export WANDB_MODE=offline   # keep wandb.init from blocking without a key

export NUM_NODES=2                          # MUST equal the job replica count
# NOTE: NO STAGE0_COMBINE here -- the fusion must stay trainable for align+recon.
export OUT_DIR_OVERRIDE=/sensei-fs-3/users/hongyangd/ckpt/alignmean-cls-k7-depthattn
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-500}"   # step-level ckpt_latest for preemption resume
export CKPT_KEEP_RECENT="${CKPT_KEEP_RECENT:-3}"

bash /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2/run_pluto_decoder_4node.sh alignmean-k7
