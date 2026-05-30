#!/usr/bin/env bash
set -euo pipefail

# Fedora-oriented v4l2loopback setup for:
# - OBS Virtual Camera on /dev/video0
# - Avatar Camera on /dev/video10
#
# Defaults favor OBS ingestion reliability for /dev/video10:
# - OBS camera (/dev/video0): exclusive_caps=1
# - Avatar camera (/dev/video10): exclusive_caps=0
# Override via environment variables if needed.

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root: sudo $0"
  exit 1
fi

OBS_VIDEO_NR="${OBS_VIDEO_NR:-0}"
AVATAR_VIDEO_NR="${AVATAR_VIDEO_NR:-10}"
OBS_EXCLUSIVE_CAPS="${OBS_EXCLUSIVE_CAPS:-1}"
AVATAR_EXCLUSIVE_CAPS="${AVATAR_EXCLUSIVE_CAPS:-0}"

VIDEO_NR="${OBS_VIDEO_NR},${AVATAR_VIDEO_NR}"
EXCLUSIVE_CAPS="${OBS_EXCLUSIVE_CAPS},${AVATAR_EXCLUSIVE_CAPS}"
CARD_LABELS="OBS Virtual Camera,Avatar Camera"

echo "[1/6] Installing dependencies..."
dnf install -y dkms kernel-devel kernel-headers v4l-utils

echo "[2/6] Enabling v4l2loopback Copr..."
dnf copr -y enable kuya-carlo/v4l2loopback || true

echo "[3/6] Installing v4l2loopback..."
dnf install -y v4l2loopback

echo "[4/6] Loading module..."
modprobe -r v4l2loopback 2>/dev/null || true
modprobe v4l2loopback devices=2 video_nr="${VIDEO_NR}" exclusive_caps="${EXCLUSIVE_CAPS}" \
  card_label="${CARD_LABELS}"

echo "[5/6] Writing persistent config..."
cat >/etc/modules-load.d/v4l2loopback.conf <<'EOF'
v4l2loopback
EOF

cat >/etc/modprobe.d/v4l2loopback.conf <<EOF
options v4l2loopback devices=2 video_nr=${VIDEO_NR} exclusive_caps=${EXCLUSIVE_CAPS} card_label="${CARD_LABELS}"
EOF

echo "[6/6] Verifying device..."
v4l2-ctl --list-devices | sed -n '1,80p'
echo "Module options:"
echo "  video_nr=${VIDEO_NR}"
echo "  exclusive_caps=${EXCLUSIVE_CAPS}"
echo "Done. Test streams with:"
echo "  ffplay /dev/video10    # Avatar Camera"
echo "  ffplay /dev/video0     # OBS Virtual Camera"
