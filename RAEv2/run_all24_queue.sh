#!/usr/bin/env bash
# ============================================================
#  K24 stage-1 queue on GPUs 0-3 (GPUs 4-7 belong to another job):
#    (wait for the running dropmean_bn_all24 to finish)
#    -> plain softgate   (raev2 + learnable gate, no proj/SIGReg/dropout)
#    -> dropmean LN, SIGReg OFF (isolates projector+dropout; LN-vs-BN twin)
#  Each script auto-resumes, so re-running this queue continues.
#
#  docker exec -d junjie_raev2 bash -c "cd /workspace/RAEv3/RAEv2 && \
#      nohup bash run_all24_queue.sh > output_full/run_all24_queue.log 2>&1"
# ============================================================
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

banner () {
    echo "##################################################################"
    echo "#####  $(date '+%F %T')  $1"
    echo "##################################################################"
}

cleanup () {
    pkill -9 -f '[t]rain_decoder_mls_softgate_plain|[t]rain_decoder_mls_dropmean_sigreg' 2>/dev/null || true
    sleep 20
}

run_one () {
    local script="$1" outdir="$2"
    mkdir -p "${outdir}"
    banner "START  ${script}"
    bash "${script}" >> "${outdir}/train.log" 2>&1
    local rc=$?
    banner "DONE   ${script} (exit ${rc})  ->  ${outdir}/train.log"
    cleanup
}

banner "waiting for dropmean_bn_all24 to finish"
while pgrep -f '[t]rain_decoder_mls_dropmean_bn' >/dev/null; do sleep 60; done

run_one run_train_decoder_softgate_all24.sh        output_full/train_decoder_mls_softgate_all24
run_one run_train_decoder_dropmean_ln_nosig_all24.sh output_full/train_decoder_mls_dropmean_ln_nosig_all24

banner "ALL DONE (K24 queue)"
