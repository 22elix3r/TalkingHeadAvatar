#!/usr/bin/env bash
# =============================================================================
# install_llama_cpp_python.sh
#
# Install llama-cpp-python with CUDA support (NVIDIA RTX 4070 Ti, CUDA 13.x).
# Run this AFTER training finishes to avoid competing for GPU memory.
#
# Usage:
#   bash scripts/install_llama_cpp_python.sh
# =============================================================================
set -euo pipefail

echo "==> Installing llama-cpp-python (CUDA backend)..."

# Try the CUDA wheel first (fast, no compilation). Use it as the primary index;
# with --extra-index-url pip may prefer the smaller CPU wheel from PyPI.
# cu124 wheels work with the CUDA 12.8 toolkit / CUDA 13-capable driver here.
pip install "llama-cpp-python==0.3.21" \
    --force-reinstall \
    --no-deps \
    --only-binary=:all: \
    --index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# If the above fails (architecture mismatch, etc.), fall back to source build:
# CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 CUDAHOSTCXX=/usr/bin/g++-14 \
# CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14" \
# FORCE_CMAKE=1 pip install "llama-cpp-python==0.3.21" \
#   --force-reinstall --no-cache-dir --no-deps --no-binary llama-cpp-python

echo ""
echo "==> Verifying install..."
python3 -c "from llama_cpp import llama_supports_gpu_offload; print('GPU offload:', llama_supports_gpu_offload())"
echo "Done."
