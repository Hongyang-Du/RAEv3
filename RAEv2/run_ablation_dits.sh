#!/usr/bin/env bash
# ============================================================
#  DECODER ABLATION: smallest DiT (DiT-B), 20 epochs, one 8-GPU node, BACK-TO-BACK.
#  Trains the SAME DiT-B on the 4 depth-attn stage-1 decoders' latents, to compare which
#  decoder gives the cleaner generative latent under a fixed small budget:
#
#    jepa-k23-cls  -> ckpt/stage1-decoder-jepa-depthattn-k23-cls   (CLS-on frozen JEPA k23)
#    jepa-k7-cls   -> ckpt/stage1-decoder-jepa-depthattn-k7-cls    (CLS-on frozen JEPA k7)
#    alignmean-k23 -> ckpt/alignmean-cls-k23-depthattn             (align-to-mean+cls k23)
#    alignmean-k7  -> ckpt/alignmean-cls-k7-depthattn             (align-to-mean+cls k7)
#
#  Model: DiTwDDTHeadIG, hidden [768,1024] depth [12,2] heads [12,16] base_model_depth 4
#         (DiT-B body + wide 2-block DDT head). global batch 1024, 20ep, gmuon, viz ON.
#  Results -> ckpt/abl-dit-b-<variant>/ , ckpt every 2 ep (keep recent 2 + every 10).
#  Each run auto-resumes per EXPERIMENT_NAME, so re-launching continues where it stopped.
#
#  latent_stats.pt: exists for k7-cls / alignmean-k23 / alignmean-k7 already; MISSING for
#  jepa-k23-cls -> this script computes it ONCE (250k samples) before that run.
#
#  Usage:
#    bash run_ablation_dits.sh                 # all 4, back-to-back
#    bash run_ablation_dits.sh jepa-k23-cls    # just one
#    WANDB_KEY=... bash run_ablation_dits.sh    # + wandb logging (entity uscgvl / omnirae)
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
export CKPT_KEEP_RECENT="${CKPT_KEEP_RECENT:-2}"   # keep most-recent N ep-*.pt
export CKPT_KEEP_EVERY="${CKPT_KEEP_EVERY:-10}"    # + every-K-epoch milestones
export WANDB_ENTITY="${WANDB_ENTITY:-uscgvl}"
export WANDB_PROJECT="${WANDB_PROJECT:-omnirae}"

NGPU="${NGPU:-8}"
CKPT_ROOT="$ROOT/ckpt"
DATA="${DATA:-/mnt/localssd/imagenet-256}"
NUM_STATS_SAMPLES="${NUM_STATS_SAMPLES:-250000}"
WANDB_FLAG=""; [ -n "${WANDB_KEY:-}" ] && WANDB_FLAG="--wandb"

[ -d "$DATA/imagenet-latents-images" ] || { echo "FATAL: imagenet-256 not staged at $DATA (expected $DATA/imagenet-latents-images)"; exit 1; }
ln -sfn "$DATA" "$REPO/data/imagenet-256"   # configs point at /mnt/localssd/imagenet-256; keep repo symlink in sync too

freeport () { ${PY} -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

# variant -> config, experiment name, latent_stats path (for the compute-if-missing step)
cfg_of ()  { case "$1" in
  jepa-k23-cls)  echo configs/stage2/training/imagenet-dinov3l-jepa-depthattn-k23-cls-ablB.yaml ;;
  jepa-k7-cls)   echo configs/stage2/training/imagenet-dinov3l-jepa-depthattn-k7-cls-ablB.yaml ;;
  alignmean-k23) echo configs/stage2/training/imagenet-dinov3l-alignmean-cls-depthattn-k23-ablB.yaml ;;
  alignmean-k7)  echo configs/stage2/training/imagenet-dinov3l-alignmean-cls-depthattn-k7-ablB.yaml ;;
esac; }
stats_of () { case "$1" in
  jepa-k23-cls)  echo "$CKPT_ROOT/stage1-decoder-jepa-depthattn-k23-cls/latent_stats.pt" ;;
  jepa-k7-cls)   echo "$CKPT_ROOT/stage1-decoder-jepa-depthattn-k7-cls/latent_stats.pt" ;;
  alignmean-k23) echo "$CKPT_ROOT/alignmean-cls-k23-depthattn/latent_stats.pt" ;;
  alignmean-k7)  echo "$CKPT_ROOT/alignmean-cls-k7-depthattn/latent_stats.pt" ;;
esac; }

run_one () {
  local variant="$1"
  local cfg; cfg="$(cfg_of "$variant")"
  [ -n "$cfg" ] || { echo "unknown variant: $variant"; return 1; }
  local name="abl-dit-b-${variant}"
  local stats; stats="$(stats_of "$variant")"
  mkdir -p "$CKPT_ROOT/$name"

  # 1) latent stats (cls_surrogate shifts the latent off N(0,I) -> DiT needs them). Only
  #    jepa-k23-cls is missing; the other three already have latent_stats.pt.
  if [ ! -f "$stats" ]; then
    echo "############ $(date '+%F %T')  [$variant] computing latent_stats -> $stats"
    mkdir -p "$(dirname "$stats")"
    ${TR} --nproc_per_node="${NGPU}" --master-port="$(freeport)" \
      scripts/stage1/compute_latent_stats.py \
      --config "$cfg" --data-dir "$DATA" \
      --output-path "$stats" --num-samples "${NUM_STATS_SAMPLES}" \
      2>&1 | tee "$CKPT_ROOT/$name/latent_stats.log"
  else
    echo "############ $(date '+%F %T')  [$variant] latent_stats present: $stats"
  fi

  # 2) train the DiT-B, 20ep, 8 GPU. Auto-resumes per EXPERIMENT_NAME.
  echo "############ $(date '+%F %T')  START $variant ($name)  cfg=$cfg"
  EXPERIMENT_NAME="$name" ${TR} --nproc_per_node="${NGPU}" --master-port="$(freeport)" \
    src/train.py --config "$cfg" --results-dir "$CKPT_ROOT" --precision bf16 ${WANDB_FLAG} \
    2>&1 | tee "$CKPT_ROOT/$name/train.log"
  echo "############ $(date '+%F %T')  DONE  $variant (exit ${PIPESTATUS[0]})  -> $CKPT_ROOT/$name/train.log"
  pkill -9 -f '[s]rc/train.py' 2>/dev/null || true
  sleep 20
}

VARIANTS=("jepa-k23-cls" "jepa-k7-cls" "alignmean-k23" "alignmean-k7")
if [ "$#" -gt 0 ] && [ "${1:-}" != "all" ]; then
  VARIANTS=("$@")
fi

echo "############ $(date '+%F %T')  ablation runner: ${VARIANTS[*]}  (DiT-B, per-config ep, ${NGPU} GPU, wandb=${WANDB_FLAG:-off})"
for v in "${VARIANTS[@]}"; do run_one "$v"; done
echo "############ $(date '+%F %T')  ALL DONE"
