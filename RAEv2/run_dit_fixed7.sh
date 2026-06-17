#!/usr/bin/env bash
# ============================================================
#  Fixed-7 DiT: train a normal DiT (no gate) on the EQUAL MEAN of the 7 layers the
#  selection run chose, decoded by the frozen plain k=23 unified decoder.
#  Generates a per-run config, recomputes latent_stats for the 7-mean, then trains.
#
#    LAYERS="1,2,4,9,15,22,23" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#        nohup bash run_dit_fixed7.sh > output_full/run_dit_fixed7.log 2>&1 &
# ============================================================
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"
: "${LAYERS:?set LAYERS=comma,separated,layers e.g. 1,2,4,9,15,22,23}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')
CONDA=/opt/conda/envs/rae; TR=${CONDA}/bin/torchrun; PY=${CONDA}/bin/python
DATA=/datasets/imagenet-256-full
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY=$(grep -A2 'api.wandb.ai' ~/.netrc | grep password | awk '{print $2}')
export WANDB_PROJECT=raev3-full
export WANDB_ENTITY=uscgvl
freeport () { ${PY} -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

TAG=$(echo "$LAYERS" | tr ',' '-')
CFG=configs/stage2/training/imagenet-dinov3l-fixed7-${TAG}.yaml
STATS=output_full/fixed7_${TAG}_latent_stats.pt
EXP=dit-fixed7-${TAG}

# materialize the per-run config from the template with the chosen 7 layers
${PY} - "$LAYERS" "$STATS" "$CFG" <<'PY'
import sys, re
layers, stats, cfg = sys.argv[1], sys.argv[2], sys.argv[3]
ll = [int(x) for x in layers.split(",")]
t = open("configs/stage2/training/imagenet-dinov3l-fixed7.yaml").read()
t = re.sub(r"\[layers=[0-9.]+\]", "[layers=" + ".".join(map(str, ll)) + "]", t)
t = re.sub(r"layers: \[[0-9, ]+\]", "layers: [" + ", ".join(map(str, ll)) + "]", t)
t = t.replace("output_full/fixed7_latent_stats.pt", stats)
open(cfg, "w").write(t)
print(f"wrote {cfg}  layers={ll}")
PY

echo "#####  $(date '+%F %T')  latent stats (7-mean) -> ${STATS}"
if [[ ! -f "${STATS}" ]]; then
    PYTORCH_ALLOC_CONF=expandable_segments:True ${TR} --nproc_per_node=${NGPU} --master-port=$(freeport) \
        scripts/stage1/compute_latent_stats.py --config "${CFG}" --data-dir "${DATA}" \
        --output-path "${STATS}" --num-samples 200000
fi

echo "#####  $(date '+%F %T')  START ${EXP}  (fixed 7 layers: ${LAYERS})"
EXPERIMENT_NAME=${EXP} PYTORCH_ALLOC_CONF=expandable_segments:True ${TR} \
    --nproc_per_node=${NGPU} --master-port=$(freeport) \
    src/train.py --config "${CFG}" --results-dir ckpts_full/stage2 --precision bf16 --wandb
echo "#####  $(date '+%F %T')  DONE ${EXP}"
