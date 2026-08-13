#!/usr/bin/env bash
# Launch abl-dit-b-mae-denoise-d512 (DiT-B, 40ep, 8 GPU) once the GPUs are free.
# Waits out the in-flight gFID generation+combine on this node, then hands off to
# run_ablation_dits.sh mae-denoise-d512 (which computes latent_stats once, then trains).
set -uo pipefail
cd /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2

echo "### $(date '+%F %T') mae-d512 launcher: waiting for all 8 GPUs to be idle (<2GB used)..."
while :; do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{c++} END{print c+0}')
  if [ "${busy:-1}" -eq 0 ]; then
    sleep 25   # settle: avoid the gap between gFID gen ending and the FID combine starting
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{c++} END{print c+0}')
    [ "${busy:-1}" -eq 0 ] && break
  fi
  sleep 30
done

echo "### $(date '+%F %T') GPUs free -> starting run_ablation_dits.sh mae-denoise-d512"
exec bash run_ablation_dits.sh mae-denoise-d512
