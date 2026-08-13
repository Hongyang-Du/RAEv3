#!/usr/bin/env bash
# ============================================================
#  DiT-B h1 ablation over 4 NODES: drop {0.0, 0.9} x block_idx {0, 5, 10, 15} = 8 runs, 40 ep each.
#
#  h1 = decoder hidden state after `block_idx`; the DiT diffuses it and decode() resumes the
#  frozen decoder from block_idx+1. block_idx is passed as a SINGLE NUMBER (config override
#  stage_1.params.block_idx=N); the base config's h1_stats_path auto-derives from it. drop-rate =
#  which frozen decoder ckpt (p00 / p09 base config). Per run: compute_h1_stats -> train to ep40
#  (full 8-GPU, auto-resume). Inference/eval need no extra code (RAEDecoderH1.decode resumes).
#
#  Allocation (8 runs across 4 nodes, balanced 2/2/2/2 -> one block depth per node):
#     node 0:  (drop 0.0, block 0),   (drop 0.9, block 0)
#     node 1:  (drop 0.0, block 5),   (drop 0.9, block 5)
#     node 2:  (drop 0.0, block 10),  (drop 0.9, block 10)
#     node 3:  (drop 0.0, block 15),  (drop 0.9, block 15)
#
#  Run ON EACH NODE (same script, different id):
#     cd /sensei-fs-3/users/hongyangd/RAEv3_oldnorm
#     setsid nohup bash run_ditb_h1_4node.sh 0 > ../ckpt/ditb-h1-node0.log 2>&1 < /dev/null &   # node 0
#     ...                                    1 ...node1...                                       # node 1
#     ...                                    2 ...node2...                                       # node 2
#     ...                                    3 ...node3...                                       # node 3
#
#  Ad-hoc single run (any block, one number):  bash run_ditb_h1_4node.sh 0.0 14
#  Show the allocation:                        bash run_ditb_h1_4node.sh
#
#  Tunables (env):  NPROC=8  DONE_EP=40  STAT_SAMPLES=50000  STAT_BATCH=256
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
DONE_EP=${DONE_EP:-40}
STAT_SAMPLES=${STAT_SAMPLES:-50000}
STAT_BATCH=${STAT_BATCH:-256}
DONE_CK=$(printf 'ep-%07d.pt' "$DONE_EP")

# per-node job lists: each entry = space-separated  drop:block  pairs (index = node id)
NODE_JOBS=(
  "0.0:0 0.9:0"     # node 0  -> block 0  (both drops)
  "0.0:5 0.9:5"     # node 1  -> block 5  (both drops)
  "0.0:10 0.9:10"   # node 2  -> block 10 (both drops)
  "0.0:15 0.9:15"   # node 3  -> block 15 (both drops)
)
declare -A DTAG=( [0.0]=p00 [0.9]=p09 )

cd "$REPO"
export DINOV3_REPO_DIR=$ROOT/dinov3_repo DINOV3_CKPT_DIR=$ROOT/pretrained_models/encoders/dinov3
export HF_HOME=$ROOT/.cache/huggingface TORCH_HOME=$ROOT/.cache/torch
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export CKPT_KEEP_RECENT=2 CKPT_KEEP_EVERY=10
export WANDB_ENTITY=uscgvl WANDB_PROJECT=omnirae
ln -sfn "$DATA" "$REPO/data/imagenet-256"
freeport(){ $PY -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

show_alloc(){
  echo "4-node allocation (drop:block):"
  for i in "${!NODE_JOBS[@]}"; do printf "  node %s:  %s\n" "$i" "${NODE_JOBS[$i]}"; done
}

# run_one <drop> <block>
run_one(){
  local drop="$1" B="$2" tag="${DTAG[$1]:-}"
  if [ -z "$tag" ]; then echo "[h1] SKIP unknown drop '$drop' (valid: ${!DTAG[*]})"; return; fi
  local cfg="$CFGD/imagenet-dinov3l-h1decoder-${tag}-cls-k23-ditb.yaml"   # block-agnostic base config
  local name="dit-b-h1-${tag}-k23-block${B}"
  local stat="$CKPT_ROOT/$name/h1_stats.pt"                              # == config's interpolated h1_stats_path
  local done_ck="$CKPT_ROOT/$name/checkpoints/$DONE_CK"
  local blkopt="stage_1.params.block_idx=${B}"                          # the single-number override

  if [ ! -f "$cfg" ]; then echo "[h1:$tag/b$B] MISSING config $cfg -> skip"; return; fi
  mkdir -p "$CKPT_ROOT/$name"
  if [ -f "$done_ck" ]; then echo "$(date '+%F %T') [h1:$tag/b$B] already at ep$DONE_EP -> skip"; return; fi

  # 1) per-(drop,block) h1 stats
  if [ ! -f "$stat" ]; then
    echo "$(date '+%F %T') [h1:$tag/b$B] computing h1 stats -> $stat"
    $TR --nproc_per_node=1 --master-port="$(freeport)" src/compute_h1_stats.py \
      --config "$cfg" --num-samples "$STAT_SAMPLES" --batch "$STAT_BATCH" --out "$stat" \
      "$blkopt" \
      > "$CKPT_ROOT/$name/h1_stats.log" 2>&1 \
      || { echo "[h1:$tag/b$B] STATS FAIL (see h1_stats.log) -> skip run"; return; }
  else
    echo "$(date '+%F %T') [h1:$tag/b$B] h1 stats present ($stat)"
  fi

  # 2) train with auto-resume until ep$DONE_EP
  local fails=0
  while [ ! -f "$done_ck" ]; do
    echo "$(date '+%F %T') [h1:$tag/b$B] launch (resume) name=$name"
    local start; start=$(date +%s)
    EXPERIMENT_NAME="$name" \
      $TR --nproc_per_node="$NPROC" --master-port="$(freeport)" src/train.py \
      --config "$cfg" --results-dir "$CKPT_ROOT" --precision bf16 \
      "$blkopt" \
      >> "$CKPT_ROOT/$name/train.log" 2>&1
    local dur=$(( $(date +%s) - start ))
    echo "$(date '+%F %T') [h1:$tag/b$B] exited after ${dur}s"
    [ -f "$done_ck" ] && break
    if [ "$dur" -lt 180 ]; then fails=$((fails+1)); else fails=0; fi
    if [ "$fails" -ge 6 ]; then echo "[h1:$tag/b$B] ABORT crash-loop (6 fast fails) -> next run"; return; fi
    sleep 20
  done
  echo "$(date '+%F %T') [h1:$tag/b$B] FINISHED ($done_ck)"
}

# ---- dispatch ------------------------------------------------------------
if [ "$#" -eq 0 ]; then show_alloc; exit 0; fi

if [ "$#" -eq 2 ]; then
  # ad-hoc single run: <drop> <block>
  echo "$(date '+%F %T') [h1] ad-hoc run drop=$1 block=$2"
  run_one "$1" "$2"
  echo "DITB_H1_ADHOC_DONE $(date '+%F %T')"
  exit 0
fi

# node dispatch: <node_id 0..3>
NODE="$1"
if ! [[ "$NODE" =~ ^[0-9]+$ ]] || [ "$NODE" -ge "${#NODE_JOBS[@]}" ]; then
  echo "usage: $0 <node_id 0..$(( ${#NODE_JOBS[@]} - 1 ))>   |   $0 <drop> <block>   |   $0 (show alloc)"
  show_alloc; exit 1
fi
echo "$(date '+%F %T') [h1] NODE $NODE jobs: ${NODE_JOBS[$NODE]}"
for job in ${NODE_JOBS[$NODE]}; do
  drop="${job%%:*}"; blk="${job##*:}"
  run_one "$drop" "$blk"
done
echo "DITB_H1_NODE${NODE}_ALL_DONE $(date '+%F %T')"
