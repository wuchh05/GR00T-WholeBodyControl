#!/usr/bin/env bash
# Install the SONIC mjlab training environment.
#
# This path is meant for machines that cannot run Isaac Sim/Isaac Lab, including
# Ascend NPU hosts where rollout stays on CPU and neural-network training will
# be moved to torch-npu.
#
# Usage:
#   bash install_scripts/install_mjlab_training.sh
#   DEVICE_BACKEND=cuda bash install_scripts/install_mjlab_training.sh
#   DEVICE_BACKEND=npu TORCH_NPU_PACKAGE='torch-npu==<version>' bash install_scripts/install_mjlab_training.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-sonic}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
DEVICE_BACKEND="${DEVICE_BACKEND:-cpu}"  # cpu | cuda | npu | none
TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.0}"
MJLAB_SPEC="${MJLAB_SPEC:-mjlab==1.5.0}"
INSTALL_DATA="${INSTALL_DATA:-0}"
RUN_CHECKS="${RUN_CHECKS:-1}"
LOG_FILE="${LOG_FILE:-install_mjlab_training.log}"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================="
echo "  SONIC mjlab training install started: $(date)"
echo "  repo:           $REPO_ROOT"
echo "  conda env:      $ENV_NAME"
echo "  python:         $PYTHON_VERSION"
echo "  device backend: $DEVICE_BACKEND"
echo "=============================================="

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found on PATH."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[INFO] Reusing conda env: $ENV_NAME"
else
    echo "[INFO] Creating conda env: $ENV_NAME"
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip setuptools wheel

case "$DEVICE_BACKEND" in
    cuda)
        echo "[INFO] Installing PyTorch $TORCH_VERSION CUDA 12.8 wheels."
        python -m pip install \
            "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
            --index-url https://download.pytorch.org/whl/cu128
        ;;
    cpu)
        echo "[INFO] Installing PyTorch $TORCH_VERSION CPU wheels."
        python -m pip install \
            "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
            --index-url https://download.pytorch.org/whl/cpu
        ;;
    npu)
        echo "[INFO] Installing PyTorch $TORCH_VERSION CPU wheels before torch-npu."
        python -m pip install \
            "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
            --index-url https://download.pytorch.org/whl/cpu
        TORCH_NPU_PACKAGE="${TORCH_NPU_PACKAGE:-torch-npu}"
        echo "[INFO] Installing torch-npu from: $TORCH_NPU_PACKAGE"
        python -m pip install "$TORCH_NPU_PACKAGE"
        ;;
    none)
        echo "[INFO] Skipping explicit torch install; existing torch will be used."
        ;;
    *)
        echo "[ERROR] DEVICE_BACKEND must be one of: cpu, cuda, npu, none"
        exit 1
        ;;
esac

echo "[INFO] Installing gear_sonic training dependencies."
python -m pip install -e "gear_sonic[training]"

echo "[INFO] Installing mjlab: $MJLAB_SPEC"
python -m pip install "$MJLAB_SPEC"

echo "[INFO] Installing local gear_sonic with mjlab extra metadata."
python -m pip install -e "gear_sonic[mjlab]"

if [ "$INSTALL_DATA" = "1" ]; then
    echo "[INFO] Downloading SONIC checkpoint and SMPL data from Hugging Face."
    python -m pip install huggingface_hub
    python download_from_hf.py --training
fi

if [ "$RUN_CHECKS" = "1" ]; then
    echo "[INFO] Running mjlab environment check."
    python check_environment.py --mjlab
    if [ "$DEVICE_BACKEND" = "npu" ]; then
        echo "[INFO] Running NPU environment check."
        python check_environment.py --npu
    fi
fi

cat <<EOF
==============================================
  SONIC mjlab training install finished: $(date)
  log: $LOG_FILE

  Activate:
    conda activate $ENV_NAME

  Optional data download:
    python download_from_hf.py --training

  Optional MotionLib PKL -> mjlab NPZ export:
    python gear_sonic/data_process/export_motionlib_robot_to_mjlab_npz.py \\
      --input data/motion_lib_bones_seed/robot_filtered \\
      --output data/motion_lib_bones_seed/mjlab_npz_motionlib_50f \\
      --device cpu --workers 16 --skip-existing
==============================================
EOF
