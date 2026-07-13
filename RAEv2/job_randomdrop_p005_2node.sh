#!/usr/bin/env bash
# 2-node (2 x 8 GPU) stage-1 decoder training with uniform random-drop at p_drop=0.05.
# Lowest point of the drop-rate sweep (alongside p0.1 / p0.5 / p0.7). Every layer kept
# with prob 0.95; z0 = equal-weight mean over the kept subset. Eval = full 23-layer mean.
#
# Pluto job: replicas = 2, GPUs/replica = 8. Paste this whole block into the job "Scripts" field.
set -exo pipefail
export WANDB_KEY=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
export WANDB_ENTITY=uscgvl
export WANDB_PROJECT=omnirae

export NUM_NODES=2
export NO_EMA=1
export OUT_DIR_OVERRIDE=/sensei-fs-3/users/hongyangd/ckpt/omni-randomdrop-plain-k23-nano-p0.05-2node
bash /sensei-fs-3/users/hongyangd/RAEv3/RAEv2/run_pluto_decoder_4node.sh nano-drop-p005
