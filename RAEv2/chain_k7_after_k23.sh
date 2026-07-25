#!/usr/bin/env bash
# Wait for the k23 5ep decoder run to finish (its torchrun/train_decoder process gone
# AND ckpt_ep005.pt written), then launch the k7 5ep decoder on the freed 8 GPUs.
set -uo pipefail
cd /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2
K23_DONE=/sensei-fs-3/users/hongyangd/ckpt/stage1-decoder-jepa-depthattn-k23-5ep/ckpt_ep005.pt
K7_LOG=/sensei-fs-3/users/hongyangd/logs/jepa-dec-5ep-k7.log
while true; do
  running=$(pgrep -af 'jepa-k23-stage1-decoder-5ep' | grep -v chain_k7 | wc -l)
  if [ "$running" -eq 0 ] && [ -f "$K23_DONE" ]; then break; fi
  sleep 30
done
echo "### $(date '+%F %T') k23 finished; launching k7" >> "$K7_LOG"
exec bash run_jepa_decoder_5ep_1node.sh k7 >> "$K7_LOG" 2>&1
