#!/usr/bin/env bash
# Self-resuming launcher for the 80-epoch h1 DiT run. train.py auto-loads the latest
# ep-*.pt via find_resume_checkpoint, so on any crash we just relaunch and continue.
# Stops when the final checkpoint (ep-0000080.pt) exists, or after MAX restarts.
cd /sensei-fs-3/users/yunfeix/RAEv3/RAEv2

CKPT_FINAL=ckpts_full/stage2/dit-h1decoder-k23/checkpoints/ep-0000080.pt
MAX=40
i=0
while true; do
  i=$((i+1))
  echo "=== [autoresume] attempt ${i} $(date '+%F %T') ==="
  env EXPERIMENT_NAME=dit-h1decoder-k23 \
      WANDB_ENTITY=uscgvl WANDB_PROJECT=raev3-full WANDB_BASE_URL=https://api.wandb.ai \
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_ALLOC_CONF=expandable_segments:True \
      .venv/bin/torchrun --nproc_per_node=8 --master-port=29536 src/train.py \
      --config configs/stage2/training/imagenet-dinov3l-h1decoder-k23.yaml \
      --results-dir ckpts_full/stage2 --precision bf16 --wandb
  rc=$?
  echo "=== [autoresume] torchrun exited rc=${rc} $(date '+%F %T') ==="
  if [ -f "${CKPT_FINAL}" ]; then
    echo "=== [autoresume] TRAINING COMPLETE (ep-0000080.pt present) ==="; break
  fi
  if [ "${i}" -ge "${MAX}" ]; then
    echo "=== [autoresume] hit MAX=${MAX} restarts, giving up ==="; break
  fi
  echo "=== [autoresume] resuming in 20s ==="; sleep 20
done
