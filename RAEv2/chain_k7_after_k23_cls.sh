#!/usr/bin/env bash
# Wait for the k23 CLS-ON 16ep decoder run to finish (its torchrun process gone AND
# ckpt_ep016.pt written), then launch the k7 CLS-ON 16ep decoder on the freed 8 GPUs.
set -uo pipefail
cd /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2
K23_DONE=/sensei-fs-3/users/hongyangd/ckpt/stage1-decoder-jepa-depthattn-k23-cls/ckpt_ep016.pt
K7_LOG=/sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2/logs/jepa-dec-cls-16ep-k7.log
while true; do
  running=$(pgrep -af 'jepa-dec-cls-16ep-k23' | grep -v chain_k7 | wc -l)
  if [ "$running" -eq 0 ] && [ -f "$K23_DONE" ]; then break; fi
  sleep 30
done
echo "### $(date '+%F %T') k23 cls finished; launching k7 cls" >> "$K7_LOG"
exec bash run_jepa_decoder_cls_16ep_1node.sh k7 >> "$K7_LOG" 2>&1
