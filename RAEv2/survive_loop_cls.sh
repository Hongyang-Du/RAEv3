#!/usr/bin/env bash
# Detached supervisor (launch me via `setsid ... < /dev/null &`) so I survive VSCode /
# claude-code exit. I keep the CLS-ON 16ep decoder alive across preemption/logout:
#   phase 1: ensure k23 is running (auto-resumes from ckpt_latest) until ckpt_ep016.pt
#   phase 2: same for k7
# Non-destructive: I only LAUNCH when nothing is already running for that variant, so I
# never duplicate a healthy job or fight for the 8 GPUs.
# Guard: if a freshly launched job dies within MIN_RUN secs, count it; after MAX_FAILS
# consecutive fast deaths (a real divergence, not a preemption) I give up on that variant.
set -uo pipefail
cd /sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2

CKPT=/sensei-fs-3/users/hongyangd/ckpt
LOGDIR=/sensei-fs-3/users/hongyangd/RAEv3_oldnorm/RAEv2/logs
SUPLOG="$LOGDIR/survive_loop_cls.log"
MIN_RUN=90        # a launch that dies faster than this counts as a "fast failure"
MAX_FAILS=5       # consecutive fast failures before giving up on a variant
POLL=30

log(){ echo "### $(date '+%F %T') supervisor: $*" >> "$SUPLOG"; }

# running? match the actual worker/torchrun cmdline (config path is unique per variant)
is_running(){ pgrep -f "train_decoder.py .*jepa-$1-stage1-decoder-cls" >/dev/null 2>&1 || \
              pgrep -f "rdzv_id=jepa-dec-cls-16ep-$1" >/dev/null 2>&1 ; }

run_phase(){
  local k="$1"
  local done_ckpt="$CKPT/stage1-decoder-jepa-depthattn-${k}-cls/ckpt_ep016.pt"
  local vlog="$LOGDIR/jepa-dec-cls-16ep-${k}.log"
  local fails=0
  log "phase $k start (done sentinel: $done_ckpt)"
  while [ ! -f "$done_ckpt" ]; do
    if ! is_running "$k"; then
      if [ "$fails" -ge "$MAX_FAILS" ]; then
        log "phase $k GIVING UP after $fails consecutive fast failures — check $vlog"
        return 1
      fi
      log "phase $k: no job running -> launching (fails so far=$fails)"
      echo "### $(date '+%F %T') supervisor launch $k" >> "$vlog"
      local t0=$(date +%s)
      bash run_jepa_decoder_cls_16ep_1node.sh "$k" >> "$vlog" 2>&1
      local dt=$(( $(date +%s) - t0 ))
      if [ -f "$done_ckpt" ]; then log "phase $k: finished (ep016 present)"; break; fi
      if [ "$dt" -lt "$MIN_RUN" ]; then fails=$((fails+1)); log "phase $k: died after ${dt}s (fast fail #$fails)";
      else fails=0; log "phase $k: exited after ${dt}s (preemption?) — will relaunch"; fi
    fi
    sleep "$POLL"
  done
  log "phase $k DONE"
  return 0
}

log "started (pid=$$, sid=$(ps -o sid= -p $$ | tr -d ' '))"
run_phase k23 && run_phase k7
log "all phases complete — exiting"
