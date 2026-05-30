#!/usr/bin/env bash
# Activate the TalkingHeadAvatar virtual environment with all required env vars.
# Usage: source activate_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/.venv/bin/activate"

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Force GCC 14 for any CUDA extension builds (system default is GCC 15, incompatible with CUDA 12.8)
export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14

echo "✓ venv activated: Python $(/home/elix3r/projects/TalkingHeadAvatar/.venv/bin/python --version)"
echo "✓ CUDA: ${CUDA_HOME}"
echo "✓ CC/CXX: gcc-14 / g++-14"
