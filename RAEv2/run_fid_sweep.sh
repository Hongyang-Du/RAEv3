#!/usr/bin/env bash
# ============================================================
#  Per-checkpoint generation FID sweep across all DiT variants.
#  One eval_fid_dit.py job per ckpt, distributed over the GPUs
#  (1 job/GPU, ~15-25 GiB each, ~15-20 min/job at 5k samples / batch 128).
#  Skips ckpts whose fid json already exists -> safe to re-run.
#
#  Run ONLY when no training is using the GPUs (guarded below).
#
#  docker exec -d rae bash -c "cd /workspace/RAEv2 && \
#      nohup bash run_fid_sweep.sh > ckpts_full/stage2/fid_sweep.log 2>&1"
#
#  Results: ckpts_full/stage2/dit-<variant>/fid/fid_ep-XXXXXXX.json
#  Curves:  plot_fid_progress.py -> ckpts_full/stage2/dit_fid_compare.png
# ============================================================
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

PYTHON=/opt/conda/envs/rae/bin/python
NGPU=8
NUM_SAMPLES=${NUM_SAMPLES:-5000}    # SAME N + seed for every point -> comparable
BATCH=${BATCH:-256}
DATA=/datasets/imagenet-256-full

# variant -> "expdir config"
declare -A RUNS=(
  [raev2mls]="ckpts_full/stage2/dit-raev2mls-k7 configs/stage2/training/imagenet-dinov3l-k7-raev2mls.yaml"
  [nogate]="ckpts_full/stage2/dit-nogate-k7 configs/stage2/training/imagenet-dinov3l-k7-nogate-sigreg.yaml"
  [dropmean]="ckpts_full/stage2/dit-dropmean-k7 configs/stage2/training/imagenet-dinov3l-k7-dropmean-sigreg.yaml"
)

if pgrep -f '[s]rc/train.py|[s]rc/train_decoder_mls|[s]rc/train_e2e' > /dev/null; then
    echo "ABORT: a training job is running — FID sweep would OOM. Wait for the queue."
    exit 1
fi

# ── collect pending jobs: "variant config ckpt outjson logfile" ──
jobs=()
for v in raev2mls nogate dropmean; do
    read -r expdir config <<< "${RUNS[$v]}"
    [[ -d "${expdir}/checkpoints" ]] || { echo "skip ${v}: no ${expdir}/checkpoints"; continue; }
    mkdir -p "${expdir}/fid"
    for ckpt in "${expdir}"/checkpoints/ep-*.pt; do
        [[ -e "${ckpt}" ]] || continue
        ep=$(basename "${ckpt}" .pt)
        out="${expdir}/fid/fid_${ep}.json"
        [[ -f "${out}" ]] && { echo "skip ${v}/${ep}: ${out} exists"; continue; }
        jobs+=("${v} ${config} ${ckpt} ${out} ${expdir}/fid/fid_${ep}.log")
    done
done
echo "$(date '+%F %T')  ${#jobs[@]} FID jobs, ${NUM_SAMPLES} samples each, ${NGPU} GPUs"

# ── run, 1 job per GPU ───────────────────────────────────────
running=0
gpu=0
for job in "${jobs[@]}"; do
    read -r v config ckpt out log <<< "${job}"
    echo "$(date '+%F %T')  GPU${gpu}  ${v}  $(basename "${ckpt}")"
    CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} src/eval_fid_dit.py \
        --config "${config}" --ckpt "${ckpt}" --data "${DATA}" \
        --num-samples "${NUM_SAMPLES}" --batch "${BATCH}" --out "${out}" \
        > "${log}" 2>&1 &
    running=$((running + 1))
    gpu=$(( (gpu + 1) % NGPU ))
    if (( running >= NGPU )); then
        wait -n
        running=$((running - 1))
    fi
done
wait

# ── summary ──────────────────────────────────────────────────
echo ""
echo "================= FID sweep results ================="
for v in raev2mls nogate dropmean; do
    read -r expdir _ <<< "${RUNS[$v]}"
    for f in "${expdir}"/fid/fid_ep-*.json; do
        [[ -e "${f}" ]] || continue
        ${PYTHON} -c "import json,sys; d=json.load(open('${f}')); print(f\"  ${v}  ep={d['epoch']}  FID={d['fid']:.3f}\")"
    done
done
echo "$(date '+%F %T')  ALL DONE  — plot: ${PYTHON} plot_fid_progress.py"
