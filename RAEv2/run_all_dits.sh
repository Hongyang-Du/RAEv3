#!/usr/bin/env bash
# ============================================================
#  Run the 3 stage-2 DiT trainings BACK-TO-BACK (all 8 GPUs each).
#  Stage-2 auto-resumes per EXPERIMENT_NAME, so re-running this
#  queue continues wherever each run left off.
#
#  stdout of each run -> ckpts_full/stage2/dit-<variant>-k7/train.log
#  Reorder / comment out the run_one lines to change the queue.
# ============================================================
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

cleanup () {
    pkill -9 -f '[s]rc/train.py' 2>/dev/null || true
    sleep 20
}

run_one () {
    local variant="$1"
    local outdir="ckpts_full/stage2/dit-${variant}-k7"
    echo "##################################################################"
    echo "#####  $(date '+%F %T')  START  dit-${variant}"
    echo "##################################################################"
    mkdir -p "${outdir}"
    bash run_train_dit.sh "${variant}" > "${outdir}/train.log" 2>&1
    echo "#####  $(date '+%F %T')  DONE   dit-${variant} (exit $?)  ->  ${outdir}/train.log"
    cleanup
}

run_one raev2mls
run_one nogate
run_one dropmean

echo "#####  $(date '+%F %T')  ALL DONE"

# ---- launch detached (survives SSH disconnect) ----
#  docker exec -d <CID> bash -c "cd /workspace/RAEv2 && mkdir -p ckpts_full/stage2 && \
#      nohup bash run_all_dits.sh > ckpts_full/stage2/run_all.log 2>&1"
#  stop all: pkill -9 -f run_all_dits.sh ; pkill -9 -f '[s]rc/train.py'
