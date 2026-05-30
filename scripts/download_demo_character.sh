#!/bin/bash
# =============================================================================
# Download GeoAvatar Pre-trained Demo Character
# =============================================================================
# Downloads pre-trained GeoAvatar weights + pre-processed SplattingAvatar
# dataset for a demo character (actor ID: 6674443 from SplattingAvatar).
#
# Sources (from official GeoAvatar README):
#   - Pretrained weights: https://drive.google.com/drive/folders/10UP3qhp9R-e63IXSlU6HrQn0YXCnBVvh
#   - SplattingAvatar 10-actor dataset: https://drive.google.com/drive/folders/11VNeA-duQ6_5d5yZl3-0VaKSTSf5Vjdn
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/demo_character"
WEIGHTS_DIR="$PROJECT_ROOT/output/demo_character"
GEOAVATAR_DIR="$PROJECT_ROOT/GaussianAvatars"  # our GeoAvatar-based implementation

echo "============================================================"
echo "  GeoAvatar Demo Character Downloader"
echo "  Project root: $PROJECT_ROOT"
echo "============================================================"
echo ""

# --------------------------
# 1. Create directories
# --------------------------
mkdir -p "$DATA_DIR"
mkdir -p "$WEIGHTS_DIR"
mkdir -p "$GEOAVATAR_DIR"

# --------------------------
# 2. Download SplattingAvatar dataset (10 actors, pre-processed in VHAP format)
#    Google Drive folder ID: 11VNeA-duQ6_5d5yZl3-0VaKSTSf5Vjdn
# --------------------------
echo "[1/2] Downloading SplattingAvatar pre-processed dataset (one actor)..."
echo "      Folder: https://drive.google.com/drive/folders/11VNeA-duQ6_5d5yZl3-0VaKSTSf5Vjdn"
echo ""

# Download the entire folder (gdown will pull all files)
cd "$DATA_DIR"
python3 -m gdown --folder "https://drive.google.com/drive/folders/11VNeA-duQ6_5d5yZl3-0VaKSTSf5Vjdn" \
    -O "$DATA_DIR"

echo ""
echo "[1/2] Dataset download complete → $DATA_DIR"

# --------------------------
# 3. Download GeoAvatar pre-trained weights
#    Google Drive folder ID: 10UP3qhp9R-e63IXSlU6HrQn0YXCnBVvh
# --------------------------
echo "[2/2] Downloading GeoAvatar pre-trained weights..."
echo "      Folder: https://drive.google.com/drive/folders/10UP3qhp9R-e63IXSlU6HrQn0YXCnBVvh"
echo ""

cd "$WEIGHTS_DIR"
python3 -m gdown --folder "https://drive.google.com/drive/folders/10UP3qhp9R-e63IXSlU6HrQn0YXCnBVvh" \
    -O "$WEIGHTS_DIR"

echo ""
echo "[2/2] Weights download complete → $WEIGHTS_DIR"

# --------------------------
# 4. Link FLAME assets into the GeoAvatar weights directory
#    (GeoAvatar render.py expects flame assets alongside the model)
# --------------------------
echo ""
echo "[3/3] Setting up FLAME asset symlinks..."

FLAME_ASSETS="$PROJECT_ROOT/GaussianAvatars/flame_model/assets/flame"
if [ -f "$FLAME_ASSETS/flame2023.pkl" ]; then
    echo "      FLAME assets found at: $FLAME_ASSETS"
    # For each downloaded model dir, link flame assets
    for model_dir in "$WEIGHTS_DIR"/*/; do
        if [ -d "$model_dir" ]; then
            model_flame_dir="$model_dir/flame_model/assets/flame"
            if [ ! -d "$model_flame_dir" ]; then
                mkdir -p "$model_flame_dir"
                # Symlink key FLAME files
                ln -sf "$FLAME_ASSETS/flame2023.pkl" "$model_flame_dir/flame2023.pkl" 2>/dev/null || true
                ln -sf "$FLAME_ASSETS/FLAME_masks.pkl" "$model_flame_dir/FLAME_masks.pkl" 2>/dev/null || true
                echo "      Linked FLAME assets → $model_flame_dir"
            fi
        fi
    done
else
    echo "      WARNING: FLAME assets not found at $FLAME_ASSETS"
    echo "               Download flame2023.pkl from https://flame.is.tue.mpg.de/download.php"
fi

# --------------------------
# 5. Print summary
# --------------------------
echo ""
echo "============================================================"
echo "  DOWNLOAD COMPLETE"
echo "============================================================"
echo ""
echo "  Dataset location:  $DATA_DIR"
echo "  Weights location:  $WEIGHTS_DIR"
echo ""
echo "  To render with GeoAvatar render.py, run:"
echo ""

# Find the first downloaded model and dataset
FIRST_MODEL=$(find "$WEIGHTS_DIR" -maxdepth 2 -name "point_cloud" -type d | head -1 | xargs dirname 2>/dev/null || echo "$WEIGHTS_DIR/<model_dir>")
FIRST_DATA=$(find "$DATA_DIR" -maxdepth 2 -name "transformsMetracker" -type d | head -1 | xargs dirname 2>/dev/null || echo "$DATA_DIR/<actor_dir>")

echo "  python $GEOAVATAR_DIR/render.py \\"
echo "    -m \"$FIRST_MODEL\" \\"
echo "    --iteration 180000 \\"
echo "    -t \"$FIRST_DATA/transformsMetracker\" \\"
echo "    --select_camera_id 0 \\"
echo "    --skip_train --skip_val"
echo ""
echo "  Or use run_demo.py (project orchestrator) once audio pipeline is ready."
echo "============================================================"
