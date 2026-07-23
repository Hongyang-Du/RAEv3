#!/usr/bin/env bash
# ImageNet-1k linear-probe top-1 accuracy for EVERY DINOv3-L/16 intermediate
# layer (1..23), one frozen-encoder forward feeding 23 independent heads.
# Data: /mnt/localssd/imagenet-256 (arrow shards, in-repo labels).
# Results: output_full/linear_probes_alllayers/results.json (+ train.log)
#
#   nohup bash run_linear_probes_alllayers.sh > ../../logs/linear_probes_alllayers.log 2>&1 &
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"
ROOT=/sensei-fs-3/users/hongyangd
PY="$ROOT/rae_env/bin/python"
TR="$ROOT/rae_env/bin/torchrun"
[ -x "$PY" ] || { echo "FATAL: portable env not found at $ROOT/rae_env"; exit 1; }

export DINOV3_REPO_DIR="$ROOT/dinov3_repo"
export DINOV3_CKPT_DIR="$ROOT/pretrained_models/encoders/dinov3"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

NGPU=8
DATA=/mnt/localssd/imagenet-256
EPOCHS=${EPOCHS:-5}
BATCH=${BATCH:-128}

MASTER_PORT=$("$PY" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "Rendezvous port: ${MASTER_PORT}"

"$TR" --nproc_per_node="${NGPU}" --master-port="${MASTER_PORT}" \
    src/train_linear_probes_alllayers.py \
    --data "${DATA}" --epochs "${EPOCHS}" --batch-size "${BATCH}"

echo "Done. Results in output_full/linear_probes_alllayers/results.json"
