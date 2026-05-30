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

# Try prebuilt wheel first (fast, no compilation).
# cu124 wheels work on CUDA 12.4+ (which covers CUDA 13 drivers).
pip install llama-cpp-python \
    --upgrade \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# If the above fails (architecture mismatch, etc.), fall back to source build:
# CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --no-cache-dir

echo ""
echo "==> Verifying install..."
python3 -c "from llama_cpp import Llama; print('llama-cpp-python OK')"
echo "Done."
