#!/usr/bin/env bash
# ============================================================
#  Sequential stage-1 decoder training on 8x A100:
#    1) p_drop=1.0   (randomdrop-plain-k23-nano-p100-oldnorm.yaml)
#    2) p_drop=0.99  (randomdrop-plain-k23-nano-p099-oldnorm.yaml)
#  Stages the ~252GB imagenet-256 latents to local SSD first if missing.
#  train_decoder.py auto-resumes from <out_dir>/ckpt_latest.pt.
# ============================================================
set -uo pipefail
cd /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2
ROOT=/sensei-fs-3/users/hongyangd

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"

CFG100=configs/stage1/decoder/randomdrop-plain-k23-nano-p100-oldnorm.yaml
CFG099=configs/stage1/decoder/randomdrop-plain-k23-nano-p099-oldnorm.yaml

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CKPT_KEEP_RECENT="${CKPT_KEEP_RECENT:-3}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-500}"
export WANDB_MODE=offline

# ---- Stage imagenet-256 latents to local SSD (~252GB, 505 shards) ----
DST=/mnt/localssd/imagenet-256/imagenet-latents-images
SRC="s3://hongyangd-raev2-backup/raev2-data/imagenet-256/imagenet-latents-images"
EXPECTED=505
mkdir -p "$DST"
have=$(ls "$DST"/data-*.arrow 2>/dev/null | wc -l)
echo "### $(date '+%F %T') staging: have ${have}/${EXPECTED} train shards on local SSD"
if [ "$have" -lt "$EXPECTED" ]; then
  echo "### $(date '+%F %T') downloading train shards with s5cmd ..."
  s5cmd cp "${SRC}/data-*.arrow" "${DST}/"
  have=$(ls "$DST"/data-*.arrow 2>/dev/null | wc -l)
  echo "### $(date '+%F %T') staging done: ${have}/${EXPECTED} shards"
fi
if [ "$have" -lt "$EXPECTED" ]; then
  echo "FATAL: only ${have}/${EXPECTED} shards staged; aborting."; exit 1
fi

run_one () {
  local tag="$1" cfg="$2"
  local port; port=$($PY -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
  echo "===== [$(date '+%F %T')] START ${tag}  nproc=${NPROC}  cfg=${cfg}  port=${port} ====="
  "$TR" --standalone --nnodes=1 --nproc_per_node="$NPROC" --master_port="$port" \
    --rdzv_id="randomdrop-${tag}" \
    src/train_decoder.py --config "$cfg"
  local rc=$?
  echo "===== [$(date '+%F %T')] DONE ${tag} (rc=${rc}) ====="
  return $rc
}

run_one p1.0  "$CFG100"; rc1=$?
run_one p0.99 "$CFG099"; rc2=$?
echo "===== [$(date '+%F %T')] ALL DONE (p1.0 rc=${rc1}, p0.99 rc=${rc2}) ====="
