#!/usr/bin/env bash
# ============================================================
#  DiT-XL x h1 DECODER-DEPTH ablation (k23), 20 epochs each.
#
#  h1 = decoder hidden state AFTER block `block_idx`; the DiT diffuses it and decode() resumes
#  the frozen decoder from block_idx+1 (with dataset-mean CLS). Sweeps block_idx ONE AFTER
#  ANOTHER; block_idx=0 is the canonical h1 and is in the default set. Per block, sequentially:
#     1) precompute h1 stats (mean/var/mean_cls) via src/compute_h1_stats.py   [1 GPU]
#     2) train the DiT to ep20 with an auto-resume loop
#  Inference/eval need no extra code: RAEDecoderH1.decode() resumes the decoder, so in-training
#  sample_every viz and downstream offline gFID work unchanged.
#
#  Usage:
#     bash run_ditxl_h1_blocksweep_k23.sh                # all: 0 7 14 21 27
#     bash run_ditxl_h1_blocksweep_k23.sh 0 14           # only these block_idxs, in order
#
#  Detached (survives disconnect):
#     cd /sensei-fs-3/users/hongyangd/RAEv3_oldnorm
#     setsid nohup bash run_ditxl_h1_blocksweep_k23.sh \
#       > ../ckpt/ditxl-h1-k23-blocksweep.log 2>&1 < /dev/null &
#
#  Tunables (env):  NPROC=8  DONE_EP=20  STAT_SAMPLES=50000  STAT_BATCH=256
#  (grad_accum_steps=2 lives in the config -> 64/GPU on 8x80GB, global batch 1024.)
# ============================================================
set -uo pipefail

ROOT=/sensei-fs-3/users/hongyangd
REPO=$ROOT/RAEv3_oldnorm/RAEv2
PY=$ROOT/rae_env/bin/python
TR=$ROOT/rae_env/bin/torchrun
CKPT_ROOT=$ROOT/ckpt
DATA=/mnt/localssd/imagenet-256
CFGD=configs/stage2/training

NPROC=${NPROC:-8}
DONE_EP=${DONE_EP:-20}
STAT_SAMPLES=${STAT_SAMPLES:-50000}
STAT_BATCH=${STAT_BATCH:-256}
DONE_CK=$(printf 'ep-%07d.pt' "$DONE_EP")

cd "$REPO"
export DINOV3_REPO_DIR=$ROOT/dinov3_repo DINOV3_CKPT_DIR=$ROOT/pretrained_models/encoders/dinov3
export HF_HOME=$ROOT/.cache/huggingface TORCH_HOME=$ROOT/.cache/torch
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export CKPT_KEEP_RECENT=2 CKPT_KEEP_EVERY=10
export WANDB_ENTITY=uscgvl WANDB_PROJECT=omnirae
ln -sfn "$DATA" "$REPO/data/imagenet-256"
freeport(){ $PY -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

BLKS=("$@")
[ ${#BLKS[@]} -eq 0 ] && BLKS=(0 7 14 21 27)

echo "$(date '+%F %T') [h1] block_idxs=${BLKS[*]}  nproc=$NPROC target=ep$DONE_EP"

run_one(){
  local B="$1"
  local cfg="$CFGD/imagenet-dinov3l-h1decoder-plain-block${B}-cls-k23-ditxl.yaml"
  local name="dit-xl-h1-plain-k23-block${B}"
  local stat="$CKPT_ROOT/$name/h1_stats.pt"
  local done_ck="$CKPT_ROOT/$name/checkpoints/$DONE_CK"

  if [ ! -f "$cfg" ]; then echo "[h1:b$B] MISSING config $cfg -> skip"; return; fi
  mkdir -p "$CKPT_ROOT/$name"
  if [ -f "$done_ck" ]; then echo "$(date '+%F %T') [h1:b$B] already at ep$DONE_EP -> skip"; return; fi

  # 1) h1 stats (mean/var/mean_cls). Single GPU as in the legacy h1 recipe.
  if [ ! -f "$stat" ]; then
    echo "$(date '+%F %T') [h1:b$B] computing h1 stats -> $stat"
    $TR --nproc_per_node=1 --master-port="$(freeport)" src/compute_h1_stats.py \
      --config "$cfg" --num-samples "$STAT_SAMPLES" --batch "$STAT_BATCH" --out "$stat" \
      > "$CKPT_ROOT/$name/h1_stats.log" 2>&1 \
      || { echo "[h1:b$B] STATS FAIL (see h1_stats.log) -> skip run"; return; }
  else
    echo "$(date '+%F %T') [h1:b$B] h1 stats present ($stat)"
  fi

  # 2) train with auto-resume until ep$DONE_EP
  local fails=0
  while [ ! -f "$done_ck" ]; do
    echo "$(date '+%F %T') [h1:b$B] launch (resume) name=$name"
    local start; start=$(date +%s)
    EXPERIMENT_NAME="$name" \
      $TR --nproc_per_node="$NPROC" --master-port="$(freeport)" src/train.py \
      --config "$cfg" --results-dir "$CKPT_ROOT" --precision bf16 \
      >> "$CKPT_ROOT/$name/train.log" 2>&1
    local dur=$(( $(date +%s) - start ))
    echo "$(date '+%F %T') [h1:b$B] exited after ${dur}s"
    [ -f "$done_ck" ] && break
    if [ "$dur" -lt 180 ]; then fails=$((fails+1)); else fails=0; fi
    if [ "$fails" -ge 6 ]; then echo "[h1:b$B] ABORT crash-loop (6 fast fails) -> next block"; return; fi
    sleep 20
  done
  echo "$(date '+%F %T') [h1:b$B] FINISHED ($done_ck)"
}

for B in "${BLKS[@]}"; do
  run_one "$B"
done
echo "DITXL_H1_K23_BLOCKSWEEP_ALL_DONE $(date '+%F %T')"
