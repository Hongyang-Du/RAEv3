#!/usr/bin/env bash
# ============================================================
#  E2E queue: both variants back-to-back, FID after each.
#    1. e2e nodrop (plain mean,        init = nogate stage-1)   ~? h
#       -> output_full/train_e2e_nodrop/   + fid.json
#    2. e2e drop   (layer-dropout 0.3, init = dropmean stage-1)
#       -> output_full/train_e2e_drop/     + fid.json
#  Training auto-resumes per OUT_DIR; FID is skipped if fid.json exists.
#  Per-epoch "Val PSNR (EMA)" (recon) + "Denoise PSNR (EMA)" in each train.log.
#
#  docker exec -d rae bash -c "cd /workspace/RAEv2 && \
#      nohup bash run_all_e2e.sh > output_full/run_all_e2e.log 2>&1"
#  stop: pkill -9 -f run_all_e2e.sh ; pkill -9 -f '[t]rain_e2e_sigreg_dit'
# ============================================================
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

PYTHON=/opt/conda/envs/rae/bin/python
FID_SAMPLES=5000        # same N/seed as the stage-2 sweep -> comparable

cleanup () {
    pkill -9 -f '[t]rain_e2e_sigreg_dit' 2>/dev/null || true
    sleep 20
}

banner () {
    echo "##################################################################"
    echo "#####  $(date '+%F %T')  $1"
    echo "##################################################################"
}

run_one () {
    local variant="$1"
    local outdir="output_full/train_e2e_${variant}"
    mkdir -p "${outdir}"

    banner "START  e2e-${variant}"
    bash run_train_e2e_sigreg_dit.sh "${variant}" >> "${outdir}/train.log" 2>&1
    local rc=$?
    banner "DONE   e2e-${variant} (exit ${rc})  ->  ${outdir}/train.log"
    cleanup
    [[ ${rc} -eq 0 ]] || { echo "e2e-${variant} failed, aborting queue"; exit ${rc}; }

    if [[ -f "${outdir}/fid.json" ]]; then
        echo "skip FID for ${variant}: ${outdir}/fid.json exists"
    else
        banner "START  FID  e2e-${variant}"
        CUDA_VISIBLE_DEVICES=0 ${PYTHON} src/eval_fid_e2e.py \
            --ckpt "${outdir}/ckpt_latest.pt" \
            --data /datasets/imagenet-256-full \
            --num-samples "${FID_SAMPLES}" \
            --grid "${outdir}/fid_grid.png" \
            --out  "${outdir}/fid.json" \
            2>&1 | tee "${outdir}/fid.log"
        banner "DONE   FID  e2e-${variant}"
    fi
}

run_one nodrop
run_one drop

# ── summary ──────────────────────────────────────────────────
echo ""
echo "================= E2E final numbers ================="
for v in nodrop drop; do
    d="output_full/train_e2e_${v}"
    echo "[e2e-${v}]"
    grep -oE 'Val PSNR \(EMA\): [0-9.]+ dB' "${d}/train.log" | tail -1 | sed 's/^/  recon /'
    grep -oE 'Denoise PSNR \(EMA\): .*' "${d}/train.log" | tail -1 | sed 's/^/  /'
    grep -oE 'FID = [0-9.]+' "${d}/fid.log" 2>/dev/null | sed 's/^/  /'
done
echo "#####  $(date '+%F %T')  ALL DONE"
