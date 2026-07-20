#!/bin/bash
# 8-GPU sharded feature extraction -> gather + per-layer probe -> branch plot.
set -e
cd /workspace/RAEv3/RAEv2
source /opt/conda/etc/profile.d/conda.sh
conda activate rae

OUT=output_full/semantic_probe
NSHARDS=8
mkdir -p $OUT

echo "=== extraction: $NSHARDS shards across GPUs 0..$((NSHARDS-1)) ==="
pids=()
for s in $(seq 0 $((NSHARDS-1))); do
  CUDA_VISIBLE_DEVICES=$s python probe_semdeg_perlayer.py --mode extract \
      --shard $s --nshards $NSHARDS --out-dir $OUT \
      > $OUT/extract_shard${s}.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
if [ $fail -ne 0 ]; then echo "!! a shard failed; check $OUT/extract_shard*.log"; exit 1; fi
echo "=== all shards done ==="

echo "=== gather + per-layer probes (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 python probe_semdeg_perlayer.py --mode probe \
    --nshards $NSHARDS --out-dir $OUT > $OUT/probe.log 2>&1

echo "=== plot (branch style) ==="
python plot_semantic_degeneration.py
echo "=== DONE ==="
