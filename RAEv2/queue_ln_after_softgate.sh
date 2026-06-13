#!/usr/bin/env bash
# waits for the softgate rerun, then launches LN-no-SIGReg on GPUs 4-7
until ! pgrep -f '[t]rain_decoder_mls_softgate_plain' >/dev/null; do sleep 60; done
cd /workspace/RAEv3/RAEv2
mkdir -p output_full/train_decoder_mls_dropmean_ln_nosig_all24
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_train_decoder_dropmean_ln_nosig_all24.sh \
    > output_full/train_decoder_mls_dropmean_ln_nosig_all24/train.log 2>&1
