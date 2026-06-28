#!/usr/bin/env bash
# A100 MULTI-NODE entry for the omnirae random-drop k23 decoder on the official general
# 4-source mix (HF-streamed + localssd cache). Separate from the H100 script: writes to a
# DIFFERENT out_dir (.../-a100) so the two runs never collide.
#
# Node-count-agnostic: per-GPU batch fixed (OOM-safe), global batch = PERGPU x total_gpus
# scales with NUM_NODES; lr sqrt-scaled to the actual global batch; 16 REAL epochs.
#
# A100 memory note: 80GB A100 fits per-GPU batch 32 (same as H100). On 40GB A100 set
# PERGPU=16 (export PERGPU=16 before launch).
#
# Pluto job: replicas = NUM_NODES (e.g. 64), GPUs/replica = 8 (A100). Scripts field:
#   export WANDB_KEY=...
#   export NUM_NODES=64
#   bash /sensei-fs-3/users/hongyangd/RAEv3/RAEv2/run_pluto_decoder_a100.sh general-4src
set -uo pipefail

for base in /sensei-fs-3 /mnt/remotes/sensei-fs-3; do
  if [ -d "$base/users/hongyangd/RAEv3/RAEv2" ]; then REPO="$base/users/hongyangd/RAEv3/RAEv2"; ROOT="$base/users/hongyangd"; break; fi
done
: "${REPO:?could not find RAEv3/RAEv2 on the sensei mount}"
cd "$REPO"

mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/${JOB_NAME:-decoder-a100}-node${RANK:-0}.log") 2>&1
echo "================ $(date '+%F %T')  host=$(hostname)  rank=${RANK:-0}  ================"

PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
[ -x "$PY" ] || { echo "FATAL: portable env not found at $ROOT/rae_env"; exit 1; }

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /mnt/localssd/raev2-wds-cache 2>/dev/null || true
export CKPT_KEEP_RECENT="${CKPT_KEEP_RECENT:-4}"
[ -n "${WANDB_KEY:-}" ] && export WANDB_API_KEY="${WANDB_KEY}"
export WANDB_ENTITY="${WANDB_ENTITY:-uscgvl}"
export WANDB_PROJECT="${WANDB_PROJECT:-omnirae}"

NUM_NODES="${NUM_NODES:?set NUM_NODES to the job replica count (e.g. 64)}"
NPROC="${NUM_OF_GPUS:-8}"
NODE_RANK="${RANK:-0}"
MASTER="${MASTER_ADDR:-${JOB_NAME}-0}"
MPORT="${MASTER_PORT:-29500}"
PERGPU="${PERGPU:-32}"                          # 80GB A100 -> 32; 40GB A100 -> set 16

case "${1:-}" in
  general-4src)
    CFG=configs/stage1/decoder/omnirae-randomdrop-k23-general-4src.yaml
    # separate out_dir from the H100 run; auto-scale lr to the actual global batch.
    export OUT_DIR_OVERRIDE="$ROOT/ckpt/omnirae-randomdrop-k23-general-4src-a100"
    eval "$("$PY" - "$NUM_NODES" "$NPROC" "$PERGPU" <<'PYEOF'
import sys, math
nodes, nproc, pergpu = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
g = pergpu * nodes * nproc
lr = 2e-4 * math.sqrt(g / 512)
print(f"export BATCH_SIZE_OVERRIDE={pergpu}")
print(f"export LR_OVERRIDE={lr:.6e}")
print(f"# global_batch={g} lr={lr:.2e} (16 real epochs; steps/epoch=69.7M/{g}={69_700_000//g})")
PYEOF
)"
    echo "### a100 4src: global=$((PERGPU*NUM_NODES*NPROC)) BATCH/GPU=$PERGPU LR=$LR_OVERRIDE epochs=16(real) out=$OUT_DIR_OVERRIDE"
    ;;
  *) echo "usage: NUM_NODES=64 bash run_pluto_decoder_a100.sh <general-4src>"; exit 1 ;;
esac

exec "$TR" \
  --nnodes="${NUM_NODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC}" \
  --rdzv_backend=c10d \
  --rdzv_id="${JOB_NAME:-decoder-a100}-4src" \
  --rdzv_endpoint="${MASTER}:${MPORT}" \
  src/train_decoder.py --config "$CFG"
