#!/usr/bin/env bash
# 4-node (4 x 8 GPU) stage-1 decoder training with a SCHEDULED random-drop rate.
# Everything else is identical to the nano random-drop k23 recipe; only p_drop is annealed
# start -> end over training. Paste this whole block into the Pluto job "Scripts" field.
# Pluto job: replicas = 4, GPUs/replica = 8.
set -exo pipefail
export WANDB_KEY=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
export WANDB_ENTITY=uscgvl
export WANDB_PROJECT=omnirae

export NUM_NODES=4
export NO_EMA=1

# ---- random-drop schedule (a -> b). Edit these to pick any range/direction. ----
export PDROP_START=0.1        # p_drop at the start of training
export PDROP_END=0.9          # p_drop at the end of training  (swap to 0.9 / 0.1 to reverse)
export PDROP_TYPE=linear      # linear | cosine
export PDROP_WARMUP_FRAC=0.0  # fraction of training held at PDROP_START before ramping

export OUT_DIR_OVERRIDE=/sensei-fs-3/users/hongyangd/ckpt/omni-randomdrop-plain-k23-nano-sched-${PDROP_START}to${PDROP_END}-4node
bash /sensei-fs-3/users/hongyangd/RAEv3/RAEv2/run_pluto_decoder_4node.sh nano-drop-sched
