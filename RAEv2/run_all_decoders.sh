#!/usr/bin/env bash
# ============================================================
#  Run the 3 MLS decoder trainings BACK-TO-BACK (sequential).
#  Each uses all 8 GPUs, so they cannot overlap: one runs to
#  completion (all procs shut + GPUs freed) before the next starts.
#  Launch detached (see command at the bottom) so you don't babysit.
#
#  Each run's stdout -> <its OUT_DIR>/train.log  (parsed by plot_val_psnr.py).
#  Reorder / comment out the run_one lines to change the queue.
# ============================================================
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

cleanup () {   # kill any lingering decoder training so the GPUs are free
    pkill -9 -f '[s]rc/train_decoder_mls' 2>/dev/null || true
    sleep 20
}

run_one () {
    local script="$1" outdir="$2"
    echo "##################################################################"
    echo "#####  $(date '+%F %T')  START  ${script}"
    echo "##################################################################"
    mkdir -p "${outdir}"
    bash "${script}" > "${outdir}/train.log" 2>&1
    local rc=$?
    echo "#####  $(date '+%F %T')  DONE   ${script}  (exit ${rc})  ->  ${outdir}/train.log"
    cleanup
}

cleanup   # stop whatever is currently running first (e.g. the old raev2)

run_one run_train_decoder_mls.sh                  output_full/train_decoder_mls_raev2
run_one run_train_decoder_mls_nogate_sigreg.sh    output_full/train_decoder_mls_nogate_sigreg
run_one run_train_decoder_mls_softgate_sigreg.sh  output_full/train_decoder_mls_softgate_sigreg
run_one run_train_decoder_mls_dropmean_sigreg.sh  output_full/train_decoder_mls_dropmean_sigreg

echo "#####  $(date '+%F %T')  ALL DONE"

# ---- launch detached (survives SSH disconnect) ----
#  docker exec -d <CID> bash -c "cd /workspace/RAEv2 && mkdir -p output_full && \
#      nohup bash run_all_decoders.sh > output_full/run_all.log 2>&1"
#  monitor: tail -f output_full/run_all.log   (which one is running)
#  stop all: pkill -9 -f run_all_decoders.sh ; pkill -9 -f '[s]rc/train_decoder_mls'
