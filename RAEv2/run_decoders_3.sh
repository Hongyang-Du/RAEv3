#!/usr/bin/env bash
# ============================================================
#  Run the 3 decoder experiments back-to-back (each uses all the
#  GPUs in CUDA_VISIBLE_DEVICES). Every 1000 optimizer steps each
#  run logs val PSNR/SSIM on a fixed random 1000-image ImageNet-val
#  subset -> <out_dir>/val_psnr_steps.tsv (+ wandb val/psnr).
#  Curves: plot_val_psnr_steps.py -> output_full/val_psnr_steps_compare.png
#
#  Both trainings auto-resume per out_dir, so re-running continues.
#
#  Launch detached (survives SSH disconnect):
#    CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash run_decoders_3.sh \
#        > output_full/run_decoders_3.log 2>&1 &
#  Stop: pkill -9 -f run_decoders_3.sh ; pkill -9 -f '[t]rain_decoder.py'
# ============================================================
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

CONFIGS=(
  configs/stage1/decoder/random-drop-layer-mls-mlp-sigreg-k23.yaml
  configs/stage1/decoder/random-drop-layer-mls-plain-k23.yaml
  configs/stage1/decoder/raev2-k23.yaml
  configs/stage1/decoder/random-drop-pfull-mls-mlp-sigreg-k23.yaml
)

banner () {
    echo "##################################################################"
    echo "#####  $(date '+%F %T')  $1"
    echo "##################################################################"
}

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    banner "START  ${name}"
    bash run_decoder.sh "${cfg}"
    rc=$?
    banner "DONE   ${name} (exit ${rc})"
    pkill -9 -f '[t]rain_decoder.py' 2>/dev/null || true
    sleep 20
done

banner "ALL 3 DONE"
echo "Plot:  python plot_val_psnr_steps.py"
