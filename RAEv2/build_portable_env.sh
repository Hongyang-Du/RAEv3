#!/usr/bin/env bash
# Build a SELF-CONTAINED conda env on the shared FS so any Pluto job (p5/H100,
# ubuntu24.04 container) can run it directly via its absolute python path — no
# in-container pip/SSO/ssh-key needed. Built once from the interactive box.
set -euo pipefail

CONDA=/opt/conda/bin/conda
ENV_PREFIX=/sensei-fs-3/users/hongyangd/rae_env
PY_VER=3.11
CU_INDEX=https://download.pytorch.org/whl/cu128

unset PIP_CONSTRAINT PIP_CONSTRAINTS || true

if [[ ! -d "${ENV_PREFIX}" ]]; then
  echo ">> creating conda env at ${ENV_PREFIX} (python ${PY_VER})"
  ${CONDA} create -y --prefix "${ENV_PREFIX}" "python=${PY_VER}"
fi
PY="${ENV_PREFIX}/bin/python"
PIP="${ENV_PREFIX}/bin/pip"

echo ">> torch/torchvision (cu128)"
${PIP} install --index-url "${CU_INDEX}" "torch==2.10.0+cu128" "torchvision==0.25.0"

echo ">> public deps"
${PIP} install \
  "accelerate==0.23.0" "datasets==4.4.2" "dictdot==1.5.1" "diffusers==0.36.0" \
  "einops==0.8.1" "hf-transfer>=0.1.9" "numpy==2.2.6" "omegaconf==2.3.0" \
  "opencv-python==4.13.0.92" "pandas==2.3.3" "pillow==12.0.0" "protobuf==6.33.2" \
  "pyyaml==6.0.3" "safetensors==0.7.0" "scipy==1.15.3" "sentencepiece==0.2.1" \
  "tabulate==0.10.0" "termcolor" "timm>=1.0.16" "torch-fidelity==0.3.0" \
  "torchmetrics==1.8.2" "tqdm==4.67.1" "transformers==4.57.1" "wandb==0.23.1" \
  "webdataset==1.0.2" "lpips==0.1.4" "matplotlib"

echo ">> gmuon (DiT optimizer) + CLIP from git"
GIT_SSH_COMMAND='ssh -o BatchMode=yes' ${PIP} install "git+ssh://git@github.com/nanovisionx/gmuon.git@main"
${PIP} install "git+https://github.com/openai/CLIP.git"

echo ">> sanity"
"${PY}" - <<'PY'
import torch, torchvision, timm, diffusers, transformers, lpips, gram_newton_schulz, einops, omegaconf
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gmuon OK; transformers", transformers.__version__)
PY
echo ">> DONE: ${ENV_PREFIX}"
