#!/usr/bin/env bash
# ============================================================
#  RAEv1 vs RAEv2 single-image overfitting comparison
#
#  Runs both experiments back-to-back on the same image and
#  same hidden_sizes, then plots a combined comparison figure.
#
#  RAEv1: DINOv2-B, C=768,  velocity prediction
#  RAEv2: DINOv3-L, C=1024, x-prediction
#
#  Both sweep: [192, 384, 576, 768, 960, 1152]
#  Expected convergence:
#    RAEv1 → d >= 768 converges
#    RAEv2 → d >= 1024 (only d=1152 in the sweep)
# ============================================================
set -euo pipefail

ROOT=$(dirname "$(realpath "$0")")
CONDA_ENV=/home/colligo/miniconda3/envs/rae
PYTHON=${CONDA_ENV}/bin/python

echo "=================================================="
echo "  RAEv1 vs RAEv2 comparison sweep"
echo "  Widths: 192 384 576 768 960 1152"
echo "  Steps: 10000 per width, 1 GPU per width"
echo "=================================================="
echo ""

# ── Step 1: RAEv1 ─────────────────────────────────────────
echo ">>> Running RAEv1 (C=768, velocity-prediction)..."
cd "${ROOT}/RAE"
bash run_overfit_sweep.sh
cd "${ROOT}"
echo ">>> RAEv1 done."
echo ""

# ── Step 2: RAEv2 ─────────────────────────────────────────
echo ">>> Running RAEv2 (C=1024, x-prediction)..."
cd "${ROOT}/RAEv2"
bash run_overfit_sweep.sh
cd "${ROOT}"
echo ">>> RAEv2 done."
echo ""

# ── Step 3: Loss curve comparison ─────────────────────────
echo ">>> Generating loss comparison figures..."
${PYTHON} "${ROOT}/plot_comparison.py" \
    --v1-dir  "${ROOT}/RAE/output/overfit_results" \
    --v2-dir  "${ROOT}/RAEv2/output/overfit_results_v2" \
    --out-dir "${ROOT}/comparison_results"

# ── Step 4: Progression grids (rows=width, cols=step) ─────
echo ">>> Generating progression grids..."
${PYTHON} "${ROOT}/make_progression_grid.py" \
    --v1-dir    "${ROOT}/RAE/output/overfit_results" \
    --v2-dir    "${ROOT}/RAEv2/output/overfit_results_v2" \
    --out-dir   "${ROOT}/comparison_results" \
    --val-every 100 \
    --num-steps 1000 \
    --cell-w    128
echo ">>> All figures saved to ${ROOT}/comparison_results/"
