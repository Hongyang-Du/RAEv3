#!/usr/bin/env bash
# ============================================================
#  ImageNet linear probes for all latent variants + DINOv3 layers.
#  11 heads trained jointly on one frozen DINOv3-L forward:
#    L11..L23 / mls_mean / raev2-combine / nogate-proj / dropmean-proj
#  ~10 min/epoch on 8 GPUs, 5 epochs ≈ 1 h.
#  Results: output_full/linear_probes/results.json (+ train.log)
#  Plot:    plot_linear_probe.py -> output_full/linear_probe_compare.png
#
#  docker exec -d rae bash -c "cd /workspace/RAEv2 && mkdir -p output_full/linear_probes && \
#      nohup bash run_linear_probes.sh > output_full/linear_probes/train.log 2>&1"
# ============================================================
set -euo pipefail

CONDA_ENV=/opt/conda/envs/rae
TORCHRUN=${CONDA_ENV}/bin/torchrun
PYTHON=${CONDA_ENV}/bin/python
cd "$(dirname "$(realpath "$0")")"

NGPU=8
DATA=/datasets/imagenet-256-full
EPOCHS=5
BATCH=128

if pgrep -f '[s]rc/train.py|[s]rc/train_decoder_mls|[e]val_fid_dit' > /dev/null; then
    echo "WARNING: other GPU jobs are running — probes may contend for memory."
fi

MASTER_PORT=$(${PYTHON} -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "Rendezvous port: ${MASTER_PORT}"

PYTORCH_ALLOC_CONF=expandable_segments:True ${TORCHRUN} --nproc_per_node=${NGPU} \
    --master-port="${MASTER_PORT}" \
    src/train_linear_probes.py \
    --data       "${DATA}" \
    --epochs     "${EPOCHS}" \
    --batch-size "${BATCH}"

echo "Done. Results in output_full/linear_probes/results.json"
