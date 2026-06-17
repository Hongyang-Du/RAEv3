#!/usr/bin/env bash
# Waits for run_dit_compare.sh to finish both DiTs, then computes generation FID
# for each on a SHARED reference set (same --num-samples/seed) and prints sigreg vs
# plain. Run detached:  nohup bash run_fid_after.sh > output_full/run_fid_after.log 2>&1 &
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"
CONDA=/opt/conda/envs/rae; PY=${CONDA}/bin/python
MLOG=output_full/run_dit_compare.log
SIG_DIR=ckpts_full/stage2/dit-k23-drop-sigreg
PLAIN_DIR=ckpts_full/stage2/dit-k23-drop-plain
SIG_CFG=configs/stage2/training/imagenet-dinov3l-k23-drop-sigreg.yaml
PLAIN_CFG=configs/stage2/training/imagenet-dinov3l-k23-drop-plain.yaml

echo "[fid-after] waiting for both DiTs to finish ($(date '+%F %T'))"
for i in $(seq 1 1200); do
    grep -q "ALL DiT DONE" "${MLOG}" 2>/dev/null && break
    if ! pgrep -f "[r]un_dit_compare.sh" >/dev/null && ! grep -q "ALL DiT DONE" "${MLOG}" 2>/dev/null; then
        echo "[fid-after] MASTER GONE without ALL DiT DONE -- aborting (check ${MLOG})"; exit 1
    fi
    sleep 60
done

SIG_CKPT=$(ls -t ${SIG_DIR}/checkpoints/ep-*.pt 2>/dev/null | head -1)
PLAIN_CKPT=$(ls -t ${PLAIN_DIR}/checkpoints/ep-*.pt 2>/dev/null | head -1)
echo "##### $(date '+%F %T')  DiTs done -- computing FID #####"
echo "  sigreg ckpt: ${SIG_CKPT}"
echo "  plain  ckpt: ${PLAIN_CKPT}"
[[ -f "${SIG_CKPT}" && -f "${PLAIN_CKPT}" ]] || { echo "[fid-after] missing ckpt"; exit 1; }

# parallel: sigreg on GPU0, plain on GPU1 (all 8 free after training)
CUDA_VISIBLE_DEVICES=0 ${PY} src/eval_fid_dit.py --config "${SIG_CFG}" --ckpt "${SIG_CKPT}" \
    --num-samples 10000 --out ${SIG_DIR}/fid.json > output_full/fid_sigreg.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 ${PY} src/eval_fid_dit.py --config "${PLAIN_CFG}" --ckpt "${PLAIN_CKPT}" \
    --num-samples 10000 --out ${PLAIN_DIR}/fid.json > output_full/fid_plain.log 2>&1 &
P2=$!
wait ${P1}; wait ${P2}

echo "############################################################"
echo "#####  FID RESULTS (10k samples, shared reference)"
echo "#####  sigreg : $(cat ${SIG_DIR}/fid.json 2>/dev/null)"
echo "#####  plain  : $(cat ${PLAIN_DIR}/fid.json 2>/dev/null)"
echo "#####  ALL FID DONE  $(date '+%F %T')"
echo "############################################################"
