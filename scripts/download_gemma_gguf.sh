#!/usr/bin/env bash
# =============================================================================
# download_gemma_gguf.sh
#
# Download the Gemma-4-E4B-IT Q4_K_M GGUF from HuggingFace.
# Requires:  huggingface-hub (pip install huggingface-hub)  OR  wget/curl.
#
# The file lands in:  models/gemma-4-e4b-it-Q4_K_M.gguf
# which is exactly where gemma_loader.py looks by default.
#
# Usage:
#   bash scripts/download_gemma_gguf.sh
#
# To use a different quant (e.g. Q5_K_M), pass it as an arg:
#   bash scripts/download_gemma_gguf.sh Q5_K_M
# =============================================================================
set -euo pipefail

QUANT="${1:-Q4_K_M}"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
FILENAME="gemma-4-e4b-it-${QUANT}.gguf"
DEST="${DEST_DIR}/${FILENAME}"

# HuggingFace repo — bartowski maintains well-tested community GGUF quants
HF_REPO="bartowski/google_gemma-4-e4b-it-GGUF"

mkdir -p "${DEST_DIR}"

if [[ -f "${DEST}" ]]; then
    echo "==> Already downloaded: ${DEST}"
    exit 0
fi

echo "==> Downloading ${FILENAME} from ${HF_REPO} ..."
echo "    Destination: ${DEST}"
echo ""

# Prefer huggingface_hub for resumable downloads
if python3 -c "import huggingface_hub" 2>/dev/null; then
    python3 - <<PYEOF
from huggingface_hub import hf_hub_download
import shutil, os

path = hf_hub_download(
    repo_id="${HF_REPO}",
    filename="${FILENAME}",
    local_dir="${DEST_DIR}",
)
print(f"Downloaded to: {path}")
PYEOF
else
    # Fallback to wget
    URL="https://huggingface.co/${HF_REPO}/resolve/main/${FILENAME}?download=true"
    wget -c --show-progress -O "${DEST}" "${URL}"
fi

echo ""
echo "==> Saved to: ${DEST}"
ls -lh "${DEST}"
echo "Done. Set GEMMA_GGUF_PATH=${DEST} if using a non-default location."
