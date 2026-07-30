#!/usr/bin/env bash
# Wait for the imagenet-256 S3->localssd sync to finish, then launch the k7
# invertibility-test decoder (invert-stage0-bandjepa-k7) on the 8 local GPUs.
set -uo pipefail
cd /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2
LOG=/sensei-fs-3/users/hongyangd/ckpt/invert-stage0-bandjepa-k7/train.log
mkdir -p /sensei-fs-3/users/hongyangd/ckpt/invert-stage0-bandjepa-k7

# 1) wait for the aws s3 sync process to exit (data fully staged)
while pgrep -f 'aws s3 sync .*imagenet-256' >/dev/null 2>&1; do sleep 15; done
echo "### $(date '+%F %T') imagenet-256 staged: $(du -sh /mnt/localssd/imagenet-256 2>/dev/null | cut -f1); launching k7 invert" >> "$LOG"

# 2) launch (env + command copied from the k23 invert config header, k7 combine ckpt)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB_MODE=offline \
STAGE0_COMBINE=/sensei-fs-3/users/hongyangd/ckpt/stage0-bandjepa-depthattn-k7/ckpt_latest.pt \
DINOV3_REPO_DIR="$PWD/../../dinov3_repo" \
DINOV3_CKPT_DIR=/sensei-fs-3/users/hongyangd/pretrained_models/encoders/dinov3 \
exec /sensei-fs-3/users/hongyangd/rae_env/bin/torchrun --nproc_per_node=8 src/train_decoder.py \
    --config configs/stage1/decoder/invert-stage0-bandjepa-k7.yaml >> "$LOG" 2>&1
