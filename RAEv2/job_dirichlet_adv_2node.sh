#!/usr/bin/env bash
# 2-node (2 x 8 GPU) stage-1 decoder training with ADVERSARIAL Dirichlet drop.
# log_alpha is trained by gradient REVERSAL (adv_lambda=1) so alpha ASCENDS the recon
# loss -> it drops the layers the decoder leans on most (forces robustness) instead of
# collapsing to keep-shallow. Per-layer rate is bounded to [0.2, 0.8] (config p_min/p_max)
# so no layer is ever ~never / ~always dropped. Mean rate pinned to p_drop=0.5.
#
# Pluto job: replicas = 2, GPUs/replica = 8. Paste this whole block into the job "Scripts" field.
set -exo pipefail
export WANDB_KEY=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
export WANDB_ENTITY=uscgvl
export WANDB_PROJECT=omnirae

export NUM_NODES=2
export NO_EMA=1
export OUT_DIR_OVERRIDE=/sensei-fs-3/users/hongyangd/ckpt/omni-dirichletdrop-adv-plain-k23-nano-p0.5-2node
bash /sensei-fs-3/users/hongyangd/RAEv3/RAEv2/run_pluto_decoder_4node.sh nano-dirdrop-adv
