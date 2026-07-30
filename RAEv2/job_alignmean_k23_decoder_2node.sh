#!/usr/bin/env bash
set -exo pipefail

# Stage-1 decoder + TRAINABLE k23 depth-attn fusion, align-to-(mean+cls). w_align=1, 16 ep.
export WANDB_KEY=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
export WANDB_ENTITY=uscgvl
export WANDB_PROJECT=omnirae
[ -z "${WANDB_KEY}" ] && export WANDB_MODE=offline

export NUM_NODES=2                          # MUST equal the job replica count (要 4 就改 4)

# NOTE: do NOT set STAGE0_COMBINE -- the fusion must stay trainable (align+recon must reach it).
export OUT_DIR_OVERRIDE=/sensei-fs-3/users/hongyangd/ckpt/alignmean-cls-k23-depthattn
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-500}"
export CKPT_KEEP_RECENT="${CKPT_KEEP_RECENT:-3}"

bash /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2/run_pluto_decoder_4node.sh alignmean-k23
