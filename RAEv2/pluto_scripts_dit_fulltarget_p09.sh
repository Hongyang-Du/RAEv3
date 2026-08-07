#!/usr/bin/env bash
set -exo pipefail

# Pluto "Scripts" field for DiT-B fulltarget p_drop=0.9 on 8 GPUs (single replica).
# Pluto job: replicas = 1, GPUs/replica = 8  ->  NUM_NODES=1.
export WANDB_KEY=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
export WANDB_ENTITY=uscgvl
export WANDB_PROJECT=omnirae

export NUM_NODES=1                          # MUST equal the job replica count (8 cards = 1 node)

bash /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2/run_pluto_dit_fulltarget_p09_multinode.sh
