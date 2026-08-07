#!/usr/bin/env bash
# ============================================================
#  DiT-XL x drop-ratio ablation sweep (k23, DECOUPLED FULL TARGET), 20 epochs each.
#
#  Runs N drop ratios ONE AFTER ANOTHER (sequential), each on all GPUs, with a
#  per-run auto-resume loop until ep20. XL analogue of the dit-b k23 fulltarget family.
#
#  Usage:
#     bash run_ditxl_dropsweep_k23.sh                 # all 7: p00 p005 p01 p03 p05 p07 p09
#     bash run_ditxl_dropsweep_k23.sh p03 p07 p09     # only these, in this order
#
#  Detached (survives disconnect):
#     cd /sensei-fs-3/users/hongyangd/RAEv3_oldnorm
#     setsid nohup bash run_ditxl_dropsweep_k23.sh p00 p03 p09 \
#       > ../ckpt/ditxl-dropsweep-k23.log 2>&1 < /dev/null &
#
#  Tunables (env):
#     NPROC=8               # GPUs to use
#     GRAD_ACCUM_OVERRIDE=2 # micro-batch = 2048 / (NPROC*ACCUM); 8x2 -> 128/gpu, global 2048
#     DONE_EP=20            # target epoch
#
#  Latent stats are DiT-size-independent, so we REUSE the already-computed dit-b
#  p* latent_stats.pt (the XL configs point at those paths). If a p's stats file is
#  missing it is computed once into the dit-b path before training.
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
GRAD_ACCUM_OVERRIDE=${GRAD_ACCUM_OVERRIDE:-2}
DONE_EP=${DONE_EP:-20}
DONE_CK=$(printf 'ep-%07d.pt' "$DONE_EP")

cd "$REPO"
export DINOV3_REPO_DIR=$ROOT/dinov3_repo DINOV3_CKPT_DIR=$ROOT/pretrained_models/encoders/dinov3
export HF_HOME=$ROOT/.cache/huggingface TORCH_HOME=$ROOT/.cache/torch
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export CKPT_KEEP_RECENT=2 CKPT_KEEP_EVERY=10
export WANDB_ENTITY=uscgvl WANDB_PROJECT=omnirae
ln -sfn "$DATA" "$REPO/data/imagenet-256"
freeport(){ $PY -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

# tag -> dotted p suffix (used in config name, ckpt dir name, and shared latent_stats path)
declare -A SUF=( [p00]=0.0 [p005]=0.05 [p01]=0.1 [p03]=0.3 [p05]=0.5 [p07]=0.7 [p09]=0.9 )

TAGS=("$@")
[ ${#TAGS[@]} -eq 0 ] && TAGS=(p00 p005 p01 p03 p05 p07 p09)

echo "$(date '+%F %T') [sweep] tags=${TAGS[*]}  nproc=$NPROC accum=$GRAD_ACCUM_OVERRIDE target=ep$DONE_EP"

run_one(){
  local tag="$1" suf="${SUF[$1]:-}"
  if [ -z "$suf" ]; then echo "[sweep] SKIP unknown tag '$tag' (valid: ${!SUF[*]})"; return; fi
  local cfg="$CFGD/imagenet-dinov3l-omni-randomdrop-fulltarget-plain-k23-${tag}-ditxl.yaml"
  local name="dit-xl-omni-randomdrop-fulltarget-plain-k23-p${suf}"
  # XL configs REUSE the dit-b latent_stats (same decoder + combine -> identical stats)
  local stat="$CKPT_ROOT/dit-b-omni-randomdrop-fulltarget-plain-k23-p${suf}/latent_stats.pt"
  local done_ck="$CKPT_ROOT/$name/checkpoints/$DONE_CK"

  if [ ! -f "$cfg" ]; then echo "[sweep:$tag] MISSING config $cfg -> skip"; return; fi
  mkdir -p "$CKPT_ROOT/$name" "$(dirname "$stat")"

  if [ -f "$done_ck" ]; then echo "$(date '+%F %T') [sweep:$tag] already at ep$DONE_EP -> skip"; return; fi

  # 1) latent stats (compute once into the shared dit-b path only if missing)
  if [ ! -f "$stat" ]; then
    echo "$(date '+%F %T') [sweep:$tag] computing latent_stats -> $stat"
    $TR --nproc_per_node="$NPROC" --master-port="$(freeport)" scripts/stage1/compute_latent_stats.py \
      --config "$cfg" --data-dir "$DATA" --output-path "$stat" --num-samples 250000 \
      > "$(dirname "$stat")/latent_stats.log" 2>&1 \
      || { echo "[sweep:$tag] STATS FAIL -> skip run"; return; }
  else
    echo "$(date '+%F %T') [sweep:$tag] latent_stats present ($stat)"
  fi

  # 2) train with auto-resume until ep$DONE_EP
  local fails=0
  while [ ! -f "$done_ck" ]; do
    echo "$(date '+%F %T') [sweep:$tag] launch (resume) name=$name"
    local start; start=$(date +%s)
    GRAD_ACCUM_OVERRIDE=$GRAD_ACCUM_OVERRIDE EXPERIMENT_NAME="$name" \
      $TR --nproc_per_node="$NPROC" --master-port="$(freeport)" src/train.py \
      --config "$cfg" --results-dir "$CKPT_ROOT" --precision bf16 \
      >> "$CKPT_ROOT/$name/train.log" 2>&1
    local dur=$(( $(date +%s) - start ))
    echo "$(date '+%F %T') [sweep:$tag] exited after ${dur}s"
    [ -f "$done_ck" ] && break
    if [ "$dur" -lt 180 ]; then fails=$((fails+1)); else fails=0; fi
    if [ "$fails" -ge 6 ]; then echo "[sweep:$tag] ABORT crash-loop (6 fast fails) -> next tag"; return; fi
    sleep 20
  done
  echo "$(date '+%F %T') [sweep:$tag] FINISHED ($done_ck)"
}

for tag in "${TAGS[@]}"; do
  run_one "$tag"
done
echo "DITXL_DROPSWEEP_K23_ALL_DONE $(date '+%F %T')"
