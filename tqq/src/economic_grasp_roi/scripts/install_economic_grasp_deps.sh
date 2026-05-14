#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${ECONOMIC_GRASP_REPO_DIR:-/home/tqq/TQQ_ws/third_party/EconomicGrasp}"
PYTHON_BIN="${ECONOMIC_GRASP_PYTHON:-/usr/bin/python3}"
CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
BUILD_JOBS="${MAX_JOBS:-1}"

if [ ! -d "$REPO_DIR" ]; then
  echo "EconomicGrasp repo not found: $REPO_DIR" >&2
  exit 1
fi

cat <<'INFO'
This script installs the heavy EconomicGrasp Python/CUDA dependencies into the
Python environment used by ROS 2 Humble on this machine.

Default local stack:
  - Python 3.10
  - CUDA 12.x
  - system /usr/bin/python3 with rclpy
  - PyTorch installed in system Python
  - MinkowskiEngine
  - pointnet2
  - knn_pytorch
  - open3d/scipy/Pillow/tqdm/pyyaml

INFO

"$PYTHON_BIN" - <<'PY'
import sys
print('Python:', sys.executable)
print('Version:', sys.version)
if sys.version_info[:2] != (3, 10):
    raise SystemExit('EconomicGrasp and ROS Humble should run with Python 3.10.')
PY

"$PYTHON_BIN" -m pip install --user scipy 'open3d>=0.8' Pillow tqdm pyyaml

"$PYTHON_BIN" - <<'PY'
try:
    import torch
    print('torch:', torch.__version__, 'cuda_available:', torch.cuda.is_available())
except Exception as exc:
    raise SystemExit(f'torch is not installed in this Python yet: {exc}')
PY

if ! "$PYTHON_BIN" - <<'PY'
import MinkowskiEngine
print('MinkowskiEngine already installed')
PY
then
  echo "Installing bundled MinkowskiEngine. This can take a while."
  (
    cd "$REPO_DIR/libs/MinkowskiEngine"
    TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST" MAX_JOBS="$BUILD_JOBS" \
      "$PYTHON_BIN" setup.py install --user --blas=openblas
  )
fi

(
  cd "$REPO_DIR/libs/pointnet2"
  TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST" "$PYTHON_BIN" setup.py install --user
)

(
  cd "$REPO_DIR/libs/knn"
  TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST" "$PYTHON_BIN" setup.py install --user
)

"$PYTHON_BIN" - <<'PY'
mods = ['torch', 'MinkowskiEngine', 'open3d', 'pointnet2', 'knn_pytorch']
for mod in mods:
    imported = __import__(mod)
    print(mod, 'OK', getattr(imported, '__file__', ''))
PY
