#!/usr/bin/env bash
set -exo pipefail

# ============================================================================
#  Pluto multi-node launcher: Stage-2 DiT on the semantic-rent JOINT fusion+decoder
#  DepthAttnCombine (softmax, GAN-into-fusion) latent from
#  ckpt/rent-k23-depthattn-softmax-ganfusion-2node/ckpt_latest.pt.
#
#  Paste this into the Pluto job "Scripts" field. Set the job's replica count = 4
#  (= NUM_NODES below) and GPUs/replica = 8.
#
#  PREREQUISITE (run ONCE on a single node before launching this job): compute the
#  latent-normalization stats the config expects, otherwise train.py crashes loading a
#  missing normalization_stat_path:
#     cd /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2
#     DINOV3_REPO_DIR=$PWD/../../dinov3_repo \
#     DINOV3_CKPT_DIR=/sensei-fs-3/users/hongyangd/pretrained_models/encoders/dinov3 \
#     /sensei-fs-3/users/hongyangd/rae_env/bin/torchrun --nproc_per_node=8 \
#       scripts/stage1/compute_latent_stats.py \
#       --config configs/stage2/training/imagenet-dinov3l-depthattn-rent-ganfusion-k23.yaml \
#       --data-dir /datasets/imagenet-256-full \
#       --output-path /sensei-fs-3/users/hongyangd/ckpt/rent-k23-depthattn-softmax-ganfusion-2node/latent_stats.pt \
#       --num-samples 250000
#  (stats land on the shared FS, so all 4 nodes read the same file.)
# ============================================================================

export WANDB_KEY=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
export WANDB_ENTITY=uscgvl
export WANDB_PROJECT=omnirae

export NUM_NODES=4                          # MUST equal the job replica count

bash /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2/run_pluto_job_4node.sh rent-ganfusion
