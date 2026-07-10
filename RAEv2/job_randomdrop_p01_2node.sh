#!/usr/bin/env bash

# Insert your code here
set -exo pipefail
export WANDB_KEY=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

export NUM_NODES=2
export NO_EMA=1
export OUT_DIR_OVERRIDE=/sensei-fs-3/users/hongyangd/ckpt/omni-randomdrop-plain-k23-nano-p0.1-2node
bash /sensei-fs-3/users/hongyangd/RAEv3/RAEv2/run_pluto_decoder_4node.sh nano-drop-p01
