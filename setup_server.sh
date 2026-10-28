#!/usr/bin/env bash
# Lab server env: Ubuntu 24.04, RTX 5060 Ti (Blackwell, sm_120), conda at /opt/miniforge3.
# torch 2.5.1+cu124 has no sm_120 kernels. cu128 torch 2.7.1 / torchvision 0.22.1 does
# (confirmed on https://download.pytorch.org/whl/cu128 for cp310; newer builds exist).
# Run from anywhere: bash setup_server.sh
set -euo pipefail

CONDA_BASE=/opt/miniforge3
ENV_NAME=ccsfuse
TORCH_VERSION=2.7.1
TV_VERSION=0.22.1
INDEX=https://download.pytorch.org/whl/cu128

cd "$(dirname "$(readlink -f "$0")")"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "conda env ${ENV_NAME} already exists; reusing it"
else
  conda create -y -n "${ENV_NAME}" python=3.10
fi
conda activate "${ENV_NAME}"

python -m pip install -U pip
python -m pip install "torch==${TORCH_VERSION}" "torchvision==${TV_VERSION}" --index-url "${INDEX}"
# requirements-ccsfuse.txt pins numpy==1.23.5 (the code uses np.int). Install it
# again after torch so a newer numpy pulled in as a dependency does not stick.
python -m pip install -r requirements-ccsfuse.txt --only-binary :all:
python -m pip install --force-reinstall --no-deps "numpy==1.23.5"

python - <<'PY'
import sys
import numpy as np
import torch
import torchvision

print("python     ", sys.version.split()[0])
print("numpy      ", np.__version__, "np.int", hasattr(np, "int"))
print("torch      ", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
archs = list(torch.cuda.get_arch_list())
print("arch list  ", ", ".join(archs))
assert np.__version__ == "1.23.5", np.__version__
assert hasattr(np, "int"), "numpy 1.23.5 should still provide np.int"
assert any(a.replace("compute_", "sm_").startswith("sm_120") for a in archs), archs
print("setup_server.sh: versions OK")
PY

echo
echo "Activate later with:"
echo "  source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
echo "If 'import cv2' fails with libGL.so.1, install the system library: sudo apt-get install -y libgl1"
echo
echo "Server training (single GPU, paper total batch 12, accumulate=1, weight_decay scale 1)."
echo "train.py defaults stay at batch 4: batch 12 does not fit the 8 GB laptop."
echo "DroneVehicle:"
echo "  python train.py --cfg models/ccsfuse/<cfg>.yaml --name <name> \\"
echo "    --batch-size 12 --workers 8 --device 0 --nbs 12"
echo "LLVIP (Adam, lr 0.0001; same hyp otherwise):"
echo "  python train.py --cfg models/ccsfuse/<cfg>.yaml --name <name> \\"
echo "    --data data/multispectral/LLVIP_ccsfuse.yaml \\"
echo "    --hyp data/hyp.ccsfuse_llvip.yaml --adam \\"
echo "    --batch-size 12 --workers 8 --device 0 --nbs 12"
