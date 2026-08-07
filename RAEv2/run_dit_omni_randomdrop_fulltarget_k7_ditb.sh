#!/usr/bin/env bash
# ============================================================
#  DiT-B on the omni-randomdrop-plain-k7-nano-p0.3 decoder, DECOUPLED FULL TARGET.
#  Same as run_dit_omni_randomdrop_k7_ditb.sh EXCEPT transport.decoupled_full_target=true:
#  x_t is built from the random-drop latent, but the FM loss regresses to the full-mean
#  latent. 40 epochs, no LR decay. The ONLY diff vs the drop run is the regression target.
#
#  latent_stats: REUSES the drop=true marginal stat already computed for the drop run
#  (same stage-1 dir), so the stats step below is skipped when it exists.
#
#  Usage:
#    bash run_dit_omni_randomdrop_fulltarget_k7_ditb.sh
#    NGPU=8 WANDB_KEY=... bash run_dit_omni_randomdrop_fulltarget_k7_ditb.sh
# ============================================================
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
NUM_STATS_SAMPLES="${NUM_STATS_SAMPLES:-250000}"
WANDB_FLAG=""; [ -n "${WANDB_KEY:-}" ] && WANDB_FLAG="--wandb"

CFG="configs/stage2/training/imagenet-dinov3l-omni-randomdrop-fulltarget-plain-k7-nano-p03-ditb.yaml"
NAME="dit-b-omni-randomdrop-fulltarget-plain-k7-nano-p0.3"
STATS="$CKPT_ROOT/omni-randomdrop-plain-k7-nano-p0.3/latent_stats.pt"
DECODER="$CKPT_ROOT/omni-randomdrop-plain-k7-nano-p0.3/ckpt_latest.pt"

[ -f "$DECODER" ] || { echo "FATAL: decoder ckpt missing: $DECODER (download from s3://hongyangd-raev2-backup/ckpt/omni-randomdrop-plain-k7-nano-p0.3/)"; exit 1; }
[ -d "$DATA/imagenet-latents-images" ] || { echo "FATAL: imagenet-256 not staged at $DATA (expected $DATA/imagenet-latents-images)"; exit 1; }
ln -sfn "$DATA" "$REPO/data/imagenet-256"
mkdir -p "$CKPT_ROOT/$NAME"

freeport () { ${PY} -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

# 1) MARGINAL latent stat under drop=true -- REUSED from the drop run (same stage-1 dir).
if [ ! -f "$STATS" ]; then
  echo "############ $(date '+%F %T')  computing latent_stats (drop=true marginal) -> $STATS"
  mkdir -p "$(dirname "$STATS")"
  ${TR} --nproc_per_node="${NGPU}" --master-port="$(freeport)" \
    scripts/stage1/compute_latent_stats.py \
    --config "$CFG" --data-dir "$DATA" \
    --output-path "$STATS" --num-samples "${NUM_STATS_SAMPLES}" \
    2>&1 | tee "$CKPT_ROOT/$NAME/latent_stats.log"
else
  echo "############ $(date '+%F %T')  latent_stats present (reused from drop run): $STATS"
fi

# 2) train DiT-B. Auto-resumes per EXPERIMENT_NAME.
echo "############ $(date '+%F %T')  START $NAME  cfg=$CFG  ngpu=$NGPU"
EXPERIMENT_NAME="$NAME" ${TR} --nproc_per_node="${NGPU}" --master-port="$(freeport)" \
  src/train.py --config "$CFG" --results-dir "$CKPT_ROOT" --precision bf16 ${WANDB_FLAG} \
  2>&1 | tee "$CKPT_ROOT/$NAME/train.log"
echo "############ $(date '+%F %T')  DONE  $NAME (exit ${PIPESTATUS[0]})  -> $CKPT_ROOT/$NAME/train.log"
