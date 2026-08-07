#!/usr/bin/env bash
# DiT-B ablation on the OFFICIAL RAEv2 k7 decoder (stage1.RAE, decoder.pt + stats.pt).
# 5th point of the abl-dit-b-* decoder comparison. Same env as run_ablation_dits.sh but
# runs train.py directly (no latent_stats precompute -- stage1.RAE reads stats.pt).
# Auto-resumes per EXPERIMENT_NAME.
set -uo pipefail

for base in /sensei-fs-3 /mnt/remotes/sensei-fs-3; do
  if [ -d "$base/users/hongyangd/RAEv3_oldnorm/RAEv2" ]; then REPO="$base/users/hongyangd/RAEv3_oldnorm/RAEv2"; ROOT="$base/users/hongyangd"; break; fi
done
: "${REPO:?could not find RAEv3_oldnorm/RAEv2 on the sensei mount}"
cd "$REPO"

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
[ -x "$PY" ] || { echo "FATAL: portable env not found at $ROOT/rae_env"; exit 1; }

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CKPT_KEEP_RECENT="${CKPT_KEEP_RECENT:-2}"
export CKPT_KEEP_EVERY="${CKPT_KEEP_EVERY:-10}"
export WANDB_ENTITY="${WANDB_ENTITY:-uscgvl}"
export WANDB_PROJECT="${WANDB_PROJECT:-omnirae}"

NGPU="${NGPU:-8}"
CKPT_ROOT="$ROOT/ckpt"
DATA="${DATA:-/mnt/localssd/imagenet-256}"
CFG="configs/stage2/training/imagenet-dinov3l-raev2official-k7-ablB.yaml"
NAME="abl-dit-b-raev2official-k7"

[ -d "$DATA/imagenet-latents-images" ] || { echo "FATAL: imagenet-256 not staged at $DATA"; exit 1; }
[ -f "$REPO/pretrained_models/stage1/imagenet/dinov3l-k7/decoder.pt" ] || { echo "FATAL: official k7 decoder.pt missing"; exit 1; }
ln -sfn "$DATA" "$REPO/data/imagenet-256"
mkdir -p "$CKPT_ROOT/$NAME"

freeport () { ${PY} -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

echo "############ $(date '+%F %T')  START $NAME  cfg=$CFG  ngpu=$NGPU"
EXPERIMENT_NAME="$NAME" ${TR} --nproc_per_node="${NGPU}" --master-port="$(freeport)" \
  src/train.py --config "$CFG" --results-dir "$CKPT_ROOT" --precision bf16 \
  2>&1 | tee "$CKPT_ROOT/$NAME/train.log"
echo "############ $(date '+%F %T')  DONE  $NAME (exit ${PIPESTATUS[0]})"
