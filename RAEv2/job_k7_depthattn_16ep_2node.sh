#!/usr/bin/env bash
# Pluto job "Scripts" field: k7 (deep-only) depth-attn stage-1 decoder, FULL 16-epoch
# anchor recipe, 2 nodes x 8 GPUs = 16 GPUs (global batch 512, per-GPU 32).
# From scratch. Auto-resumes from <out_dir>/ckpt_latest.pt on preemption.
# Uses the RAEv3_oldnorm worktree (its own code + configs), matching the k23 depth-attn run.
set -exo pipefail

# wandb: paste the same key used by job_randomdrop_p07_2node.sh, or leave unset to log offline.
export WANDB_KEY="${WANDB_KEY:-}"
export WANDB_ENTITY=uscgvl
export WANDB_PROJECT=raev3-full

export NUM_NODES=2                        # MUST equal the job replica count
# out_dir already set in the config; override kept explicit for clarity/safety.
export OUT_DIR_OVERRIDE=/sensei-fs-3/users/hongyangd/ckpt/omni-randomdrop-plain-k7-nano-p0.3-depthattn

bash /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2/run_pluto_decoder_4node.sh nano-drop-k7-p03-depthattn
