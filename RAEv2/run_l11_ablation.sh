#!/usr/bin/env bash
# ============================================================
#  L11-ONLY control experiment, end to end (all 8 GPUs):
#    1. stage-1 decoder on DINOv3-L layer 11 only  (raev2 recipe)
#         -> output_full/train_decoder_mls_l11/   (per-epoch Val PSNR in train.log)
#    2. stage-2 DiT on that latent                 (raev2 recipe; stats auto-computed)
#         -> ckpts_full/stage2/dit-l11/           (per-epoch denoise PSNR in train.log)
#    3. generation FID, 10k samples vs 10k real    (1 GPU)
#         -> ckpts_full/stage2/dit-l11/fid.json
#
#  Both trainings auto-resume, so re-running this script continues
#  wherever it left off. Skips stage-1 if its final ckpt already exists.
#
#  ---- launch detached (survives SSH disconnect) ----
#  docker exec -d <CID> bash -c "cd /workspace/RAEv2 && \
#      nohup bash run_l11_ablation.sh > output_full/run_l11.log 2>&1"
#  stop: pkill -9 -f run_l11_ablation.sh ; pkill -9 -f '[s]rc/train'
# ============================================================
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

PYTHON=/opt/conda/envs/rae/bin/python
S1_DIR=output_full/train_decoder_mls_l11
S2_DIR=ckpts_full/stage2/dit-l11
S1_EPOCHS=5     # must match EPOCHS in run_train_decoder_mls_l11.sh

cleanup () {
    pkill -9 -f '[s]rc/train_decoder_mls' 2>/dev/null || true
    pkill -9 -f '[s]rc/train.py'          2>/dev/null || true
    sleep 20
}

banner () {
    echo "##################################################################"
    echo "#####  $(date '+%F %T')  $1"
    echo "##################################################################"
}

# ── 1. stage-1 decoder (L11 only) ────────────────────────────
s1_done=$(${PYTHON} - <<EOF 2>/dev/null || echo 0
import torch
print(int(torch.load("${S1_DIR}/ckpt_latest.pt", map_location="cpu", weights_only=False)["epoch"] >= ${S1_EPOCHS}))
EOF
)
if [[ "${s1_done}" == "1" ]]; then
    banner "SKIP   decoder-l11 (already at >= ${S1_EPOCHS} epochs)"
else
    banner "START  decoder-l11"
    mkdir -p "${S1_DIR}"
    bash run_train_decoder_mls_l11.sh >> "${S1_DIR}/train.log" 2>&1
    rc=$?
    banner "DONE   decoder-l11 (exit ${rc})  ->  ${S1_DIR}/train.log"
    cleanup
    [[ ${rc} -eq 0 ]] || { echo "decoder training failed, aborting"; exit ${rc}; }
fi

# ── 2. stage-2 DiT (latent stats computed inside if missing) ─
banner "START  dit-l11"
mkdir -p "${S2_DIR}"
bash run_train_dit.sh l11 >> "${S2_DIR}/train.log" 2>&1
rc=$?
banner "DONE   dit-l11 (exit ${rc})  ->  ${S2_DIR}/train.log"
cleanup
[[ ${rc} -eq 0 ]] || { echo "DiT training failed, aborting"; exit ${rc}; }

# ── 3. generation FID on the last checkpoint ─────────────────
CKPT=$(ls -1 "${S2_DIR}"/checkpoints/*.pt 2>/dev/null | sort | tail -1)
[[ -n "${CKPT}" ]] || { echo "no DiT checkpoint found in ${S2_DIR}/checkpoints"; exit 1; }
banner "START  FID  (${CKPT})"
CUDA_VISIBLE_DEVICES=0 ${PYTHON} src/eval_fid_dit.py \
    --config configs/stage2/training/imagenet-dinov3l-l11-raev2mls.yaml \
    --ckpt   "${CKPT}" \
    --data   /datasets/imagenet-256-full \
    --num-samples 10000 \
    --grid   "${S2_DIR}/samples/fid_grid.png" \
    --out    "${S2_DIR}/fid.json" \
    2>&1 | tee "${S2_DIR}/fid.log"
banner "DONE   FID"

# ── summary ──────────────────────────────────────────────────
echo ""
echo "================= L11 control: final numbers ================="
echo "Decoder Val PSNR (per epoch):"
grep -oE 'Val PSNR \(EMA\): [0-9.]+ dB' "${S1_DIR}/train.log" | sed 's/^/  /'
echo "DiT denoise PSNR (last epoch):"
grep -oE 'Denoise PSNR \(EMA\): .*' "${S2_DIR}/train.log" | tail -1 | sed 's/^/  /'
echo "Generation FID:"
grep -oE 'FID = .*' "${S2_DIR}/fid.log" | sed 's/^/  /'
echo "#####  $(date '+%F %T')  ALL DONE"
