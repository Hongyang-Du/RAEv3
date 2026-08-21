#!/usr/bin/env bash
# CFG guidance-scale sweep + final FID for the 80-epoch dit-h1decoder-k23 run.
# Stage A: sweep 8 scales, one per GPU, each generates a full 10k set + its own FID
#          (shared seed-42 reference, so the 10k FIDs are mutually comparable).
# Stage B: pick the FID-optimal scale, run the final 50k via 8-GPU sharded generation.
#   nohup bash run_fid_cfg.sh > output_full/fid_cfg.log 2>&1 &
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

PY=.venv/bin/python
CFG=configs/stage2/training/imagenet-dinov3l-h1decoder-k23.yaml
CKPT=ckpts_full/stage2/dit-h1decoder-k23/checkpoints/ep-0000080.pt
DATA=data/imagenet-256
OUTDIR=ckpts_full/stage2/dit-h1decoder-k23
SWEEPDIR=${OUTDIR}/cfg_sweep
SHARDDIR=${OUTDIR}/fid_shards_cfg
SEED=42
SWEEP_N=10000
FINAL_N=50000
SCALES=(1.0 1.25 1.5 1.75 2.0 2.25 2.5 3.0)   # 8 scales -> 8 GPUs
mkdir -p "${SWEEPDIR}" "${SHARDDIR}"

############################  STAGE A: 10k sweep  ############################
echo "########## $(date '+%F %T')  CFG SWEEP (${SWEEP_N} samples, one scale per GPU) ##########"
pids=()
for g in "${!SCALES[@]}"; do
    s=${SCALES[$g]}
    echo "[sweep] GPU ${g}: cfg-scale ${s}"
    CUDA_VISIBLE_DEVICES=${g} ${PY} src/eval_fid_dit.py \
        --config "${CFG}" --ckpt "${CKPT}" --data "${DATA}" \
        --num-samples ${SWEEP_N} --seed ${SEED} --cfg-scale ${s} --num-workers 4 \
        --out "${SWEEPDIR}/fid_cfg_${s}.json" \
        > "${SWEEPDIR}/sweep_scale_${s}.log" 2>&1 &
    pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -ne 0 ] && echo "[sweep] WARNING: a sweep job failed -- check ${SWEEPDIR}/sweep_scale_*.log"

echo "########## $(date '+%F %T')  SWEEP RESULTS (${SWEEP_N}-sample FID) ##########"
${PY} - "$SWEEPDIR" "${SWEEPDIR}/.best_scale" <<'PY'
import glob, json, sys
sweepdir, bestfile = sys.argv[1], sys.argv[2]
rows = []
for f in glob.glob(f"{sweepdir}/fid_cfg_*.json"):
    try:
        d = json.load(open(f)); rows.append((float(d["cfg_scale"]), d["fid"]))
    except Exception as e:
        print(f"  (skip {f}: {e})")
rows.sort(key=lambda r: r[0])
for sc, fid in rows:
    print(f"  scale={sc:<5} FID(10k)={fid:.3f}")
best = min(rows, key=lambda r: r[1])
print(f"BEST scale={best[0]} FID(10k)={best[1]:.3f}")
open(bestfile, "w").write(str(best[0]))
PY
BEST_SCALE=$(cat "${SWEEPDIR}/.best_scale")
echo "########## best CFG scale = ${BEST_SCALE} ##########"
[ -z "${BEST_SCALE}" ] && { echo "[fatal] no best scale (all sweep jobs failed?)"; exit 1; }

############################  STAGE B: final 50k  ############################
echo "########## $(date '+%F %T')  FINAL FID (${FINAL_N} samples, 8-GPU sharded, cfg-scale ${BEST_SCALE}) ##########"
NGPU=8; per=$(( FINAL_N / NGPU ))
pids=()
for g in $(seq 0 $((NGPU-1))); do
    s0=$(( g * per )); s1=$(( (g+1) * per )); [ "$g" -eq $((NGPU-1)) ] && s1=${FINAL_N}
    GRID=""; [ "$g" -eq 0 ] && GRID="--grid ${OUTDIR}/fid_grid_cfg.png"
    CUDA_VISIBLE_DEVICES=${g} ${PY} src/eval_fid_dit.py \
        --config "${CFG}" --ckpt "${CKPT}" --data "${DATA}" \
        --num-samples ${FINAL_N} --seed ${SEED} --cfg-scale ${BEST_SCALE} \
        --shard-start ${s0} --shard-end ${s1} \
        --gen-out "${SHARDDIR}/shard_${g}.npy" ${GRID} \
        > "${SHARDDIR}/gen_gpu${g}.log" 2>&1 &
    pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -ne 0 ] && { echo "[final] ERROR: a shard failed -- check ${SHARDDIR}/gen_gpu*.log"; exit 1; }

SHARDS=$(printf "${SHARDDIR}/shard_%d.npy," $(seq 0 $((NGPU-1)))); SHARDS=${SHARDS%,}
CUDA_VISIBLE_DEVICES=0 ${PY} src/eval_fid_dit.py \
    --config "${CFG}" --ckpt "${CKPT}" --data "${DATA}" \
    --num-samples ${FINAL_N} --seed ${SEED} --cfg-scale ${BEST_SCALE} \
    --pool "${SHARDS}" --out "${OUTDIR}/fid_cfg.json"

echo "############################################################"
echo "#####  CFG FID DONE $(date '+%F %T')"
echo "#####  best scale ${BEST_SCALE} | 50k result: $(cat ${OUTDIR}/fid_cfg.json 2>/dev/null)"
echo "#####  no-guidance 50k baseline: $(cat ${OUTDIR}/fid.json 2>/dev/null | tr -d '\n')"
echo "############################################################"
