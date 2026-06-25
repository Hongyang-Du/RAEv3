#!/usr/bin/env bash
# Run the 3 OmniRAE DiT experiments BACK-TO-BACK on one 8-GPU node.
# Each auto-resumes per EXPERIMENT_NAME, so re-running continues where it left off.
# Per-run stdout -> /sensei-fs-3/users/hongyangd/ckpt/<name>/train.log
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

CKPT_ROOT=/sensei-fs-3/users/hongyangd/ckpt

run_one () {
  local exp="$1" name="$2"
  mkdir -p "${CKPT_ROOT}/${name}"
  echo "############ $(date '+%F %T')  START ${exp} (${name})"
  bash run_omnirae_dits.sh "${exp}" > "${CKPT_ROOT}/${name}/train.log" 2>&1
  echo "############ $(date '+%F %T')  DONE  ${exp} (exit $?)  -> ${CKPT_ROOT}/${name}/train.log"
  pkill -9 -f '[s]rc/train.py' 2>/dev/null || true
  sleep 20
}

run_one exp2 omnirae-dit-sigreg-cls-k23
run_one exp1 omnirae-dit-h1-plain-cls-k23
run_one exp3 omnirae-dit-encoder-cls-k23
echo "############ $(date '+%F %T')  ALL DONE"
