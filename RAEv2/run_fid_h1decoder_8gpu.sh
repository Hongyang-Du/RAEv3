#!/usr/bin/env bash
# Multi-GPU generation FID for the 80-epoch dit-h1decoder-k23 run.
# 8 GPUs each generate an equal contiguous slice of the global sample set
# (shared label mapping, per-shard noise seed), then one pool pass builds the
# shared reference once and computes a single FID. ~8x faster than single-GPU.
#   nohup bash run_fid_h1decoder_8gpu.sh > output_full/fid_h1decoder_k23_8gpu.log 2>&1 &
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

PY=.venv/bin/python
CFG=configs/stage2/training/imagenet-dinov3l-h1decoder-k23.yaml
CKPT=ckpts_full/stage2/dit-h1decoder-k23/checkpoints/ep-0000080.pt
DATA=data/imagenet-256
OUTDIR=ckpts_full/stage2/dit-h1decoder-k23
SHARDDIR=${OUTDIR}/fid_shards
N=50000
NGPU=8
SEED=42
mkdir -p "${SHARDDIR}"

per=$(( N / NGPU ))                       # 6250 with N=50000, NGPU=8
echo "[8gpu-fid] $(date '+%F %T') N=${N} per-shard=${per} on ${NGPU} GPUs"

pids=()
for g in $(seq 0 $((NGPU-1))); do
    s0=$(( g * per ))
    s1=$(( (g+1) * per ))
    [ "$g" -eq $((NGPU-1)) ] && s1=${N}    # last shard absorbs remainder
    GRID=""
    [ "$g" -eq 0 ] && GRID="--grid ${OUTDIR}/fid_grid.png"
    echo "[8gpu-fid] GPU ${g}: samples [${s0}, ${s1})"
    CUDA_VISIBLE_DEVICES=${g} ${PY} src/eval_fid_dit.py \
        --config "${CFG}" --ckpt "${CKPT}" --data "${DATA}" \
        --num-samples ${N} --seed ${SEED} \
        --shard-start ${s0} --shard-end ${s1} \
        --gen-out "${SHARDDIR}/shard_${g}.npy" ${GRID} \
        > "${SHARDDIR}/gen_gpu${g}.log" 2>&1 &
    pids+=($!)
done

echo "[8gpu-fid] waiting for ${#pids[@]} generation shards..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" -ne 0 ]; then
    echo "[8gpu-fid] ERROR: a shard failed -- check ${SHARDDIR}/gen_gpu*.log"; exit 1
fi
echo "[8gpu-fid] all shards done $(date '+%F %T')"

# pool + single FID on GPU 0
SHARDS=$(printf "${SHARDDIR}/shard_%d.npy," $(seq 0 $((NGPU-1))))
SHARDS=${SHARDS%,}
echo "[8gpu-fid] pooling -> FID"
CUDA_VISIBLE_DEVICES=0 ${PY} src/eval_fid_dit.py \
    --config "${CFG}" --ckpt "${CKPT}" --data "${DATA}" \
    --num-samples ${N} --seed ${SEED} \
    --pool "${SHARDS}" --out "${OUTDIR}/fid.json"

echo "############################################################"
echo "#####  FID RESULT (${N} samples, 8-GPU sharded generation)"
echo "#####  $(cat ${OUTDIR}/fid.json 2>/dev/null)"
echo "#####  DONE $(date '+%F %T')"
echo "############################################################"
