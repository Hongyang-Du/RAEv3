#!/usr/bin/env bash
# ============================================================
#  Set up the `rae` conda env inside the training container.
#  Installs the public deps needed by run_train_attnres.sh.
#  (Private nanovisionx eval deps are skipped — eval only,
#   not used by decoder training.)
#  Run INSIDE the container:  bash setup_rae_env.sh
# ============================================================
set -euo pipefail

CONDA_DIR=/opt/conda
ENV_NAME=rae
PY_VER=3.11
CU_INDEX=https://download.pytorch.org/whl/cu128

# drop the NGC container's torch pin (/etc/pip/constraint.txt) so our
# torch==2.10.0+cu128 wheel can install in the rae env.
unset PIP_CONSTRAINT PIP_CONSTRAINTS || true

# ── 1. miniconda (idempotent) ────────────────────────────────
if [[ ! -x "${CONDA_DIR}/bin/conda" ]]; then
    echo ">> installing miniconda to ${CONDA_DIR}"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
    bash /tmp/mc.sh -b -p "${CONDA_DIR}"
    rm -f /tmp/mc.sh
fi
export PATH="${CONDA_DIR}/bin:${PATH}"

# ── 2. env (idempotent) ──────────────────────────────────────
# accept channel Terms of Service (required by recent conda)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    || true
if [[ ! -d "${CONDA_DIR}/envs/${ENV_NAME}" ]]; then
    echo ">> creating conda env ${ENV_NAME} (python ${PY_VER})"
    conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi
PIP="${CONDA_DIR}/envs/${ENV_NAME}/bin/pip"

# ── 3. torch + torchvision (CUDA 12.8 wheels) ────────────────
echo ">> installing torch/torchvision (cu128)"
${PIP} install --index-url "${CU_INDEX}" "torch==2.10.0+cu128" "torchvision==0.25.0"

# ── 4. remaining public deps (from pyproject.toml) ───────────
echo ">> installing remaining deps"
${PIP} install \
    "accelerate==0.23.0" "datasets==4.4.2" "dictdot==1.5.1" "diffusers==0.36.0" \
    "einops==0.8.1" "hf-transfer>=0.1.9" "numpy==2.2.6" "omegaconf==2.3.0" \
    "opencv-python==4.13.0.92" "pandas==2.3.3" "pillow==12.0.0" "protobuf==6.33.2" \
    "pyyaml==6.0.3" "safetensors==0.7.0" "scipy==1.15.3" "sentencepiece==0.2.1" \
    "tabulate==0.10.0" "termcolor" "timm>=1.0.16" "torch-fidelity==0.3.0" \
    "torchmetrics==1.8.2" "tqdm==4.67.1" "transformers==4.57.1" "wandb==0.23.1" \
    "webdataset==1.0.2" "lpips==0.1.4" "matplotlib"
${PIP} install "git+https://github.com/openai/CLIP.git"

# ── 5. sanity check ──────────────────────────────────────────
echo ">> versions:"
"${CONDA_DIR}/envs/${ENV_NAME}/bin/python" - <<'PY'
import torch, torchvision, timm, diffusers, transformers, lpips
print("torch       ", torch.__version__, "cuda", torch.version.cuda)
print("torchvision ", torchvision.__version__)
print("timm        ", timm.__version__)
print("diffusers   ", diffusers.__version__)
print("transformers", transformers.__version__)
print("cuda avail  ", torch.cuda.is_available(), "n_gpu", torch.cuda.device_count())
PY
echo ">> DONE. env ready at ${CONDA_DIR}/envs/${ENV_NAME}"
