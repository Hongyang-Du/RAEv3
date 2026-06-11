#!/usr/bin/env bash
# ============================================================
#  Continuously refresh ONE Val-PSNR comparison image from the
#  3 experiments' train.logs (overwrites the same PNG each cycle).
#  All three runs land in a single always-current figure.
#
#    image : output_full/val_psnr_compare.png
#    source: output_full/train_decoder_mls_{sigreg,raev2,kl}/train.log
#
#  Launch detached (see bottom). Stop: pkill -9 -f watch_val_psnr.sh
# ============================================================
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"
PYTHON=/opt/conda/envs/rae/bin/python
INTERVAL="${1:-60}"          # seconds between refreshes

echo "watch_val_psnr: refreshing output_full/val_psnr_compare.png every ${INTERVAL}s"
while true; do
    "${PYTHON}" plot_val_psnr.py 2>&1 | sed 's/^/  /'
    sleep "${INTERVAL}"
done

# ---- launch detached (survives SSH disconnect) ----
#  docker exec -d <CID> bash -c "cd /workspace/RAEv2 && \
#      nohup bash watch_val_psnr.sh 60 > output_full/watch.log 2>&1"
#  view image: output_full/val_psnr_compare.png   (re-open to refresh)
#  stop: pkill -9 -f watch_val_psnr.sh
