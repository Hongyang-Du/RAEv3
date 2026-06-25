#!/usr/bin/env bash
# Sequential single-node 8xH100 training: plain random-drop+cls decoder (30ep,
# batch 32 x grad_accum 2), THEN bn-sigreg decoder (30ep, batch 32, no accum).
# After each run finishes, the final ckpt is uploaded to S3.
#
#   nohup bash run_decoders_sequential.sh > run_decoders_sequential.log 2>&1 &
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')
PY=./.venv/bin/python
TR=./.venv/bin/torchrun
unset VIRTUAL_ENV   # don't let /opt/venv shadow the project .venv

export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/mnt/localssd/.cache/torch
export HF_HOME=/mnt/localssd/.cache/huggingface
export HF_TOKEN="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"
export WANDB_API_KEY="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"
export WANDB_ENTITY=uscgvl
export WANDB_PROJECT=raev3-full

freeport () { ${PY} -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

PLAIN_CFG=configs/stage1/decoder/random-drop-layer-mls-plain-cls-k23.yaml
SIGREG_CFG=configs/stage1/decoder/random-drop-layer-mls-mlp-sigreg-cls-k23.yaml
PLAIN_OUT=output_full/decoder_random_drop_layer_mls_plain_cls_k23_16ep_gan8
SIGREG_OUT=output_full/decoder_random_drop_layer_mls_mlp_sigreg_k23

# S3 upload target (bucket region resolved at upload time).
export AWS_PROFILE=raev3
S3_DEST=s3://hongyang-du/raev3_decoders
S3_REGION=ap-southeast-2

run_one () {  # $1=name $2=config $3=out_dir
    local name=$1 cfg=$2 out=$3
    echo "############################################################"
    echo "##### $(date '+%F %T')  TRAIN ${name}  cfg=${cfg}"
    echo "############################################################"
    PYTORCH_ALLOC_CONF=expandable_segments:True ${TR} \
        --nproc_per_node=${NGPU} --master-port=$(freeport) \
        src/train_decoder.py --config ${cfg}
    local rc=$?
    if [ ${rc} -ne 0 ]; then
        echo "##### $(date '+%F %T')  ${name} FAILED (rc=${rc}) — stopping."
        return ${rc}
    fi
    # upload final ckpt to S3 (latest + last epoch archive)
    local last_ep
    last_ep=$(ls -1 ${out}/ckpt_ep*.pt 2>/dev/null | sort | tail -1)
    echo "##### $(date '+%F %T')  ${name} DONE — uploading final ckpt to ${S3_DEST}/${name}/"
    aws s3 cp "${out}/ckpt_latest.pt" "${S3_DEST}/${name}/ckpt_latest.pt" --region "${S3_REGION}" \
        && echo "  uploaded ${name}/ckpt_latest.pt" || echo "  WARN: upload ckpt_latest failed"
    if [ -n "${last_ep}" ]; then
        aws s3 cp "${last_ep}" "${S3_DEST}/${name}/$(basename ${last_ep})" --region "${S3_REGION}" \
            && echo "  uploaded ${name}/$(basename ${last_ep})" || echo "  WARN: upload ${last_ep} failed"
    fi
    return 0
}

echo "===== Sequential decoder training on ${NGPU} GPUs ====="
run_one plain  "${PLAIN_CFG}"  "${PLAIN_OUT}"  || { echo "PLAIN failed; not starting sigreg."; exit 1; }
run_one sigreg "${SIGREG_CFG}" "${SIGREG_OUT}" || { echo "SIGREG failed."; exit 1; }

echo "############################################################"
echo "#####  $(date '+%F %T')  BOTH DECODERS DONE"
echo "#####  Final ckpts uploaded to ${S3_DEST}/{plain,sigreg}/"
echo "############################################################"
