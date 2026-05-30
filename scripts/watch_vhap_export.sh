#!/usr/bin/env bash
set -euo pipefail

LOG_PATH="${1:?usage: watch_vhap_export.sh LOG_PATH}"
PROJECT_ROOT="/home/elix3r/projects/TalkingHeadAvatar"
TRACK_SESSION="vhap_track"
SRC_FOLDER="$PROJECT_ROOT/output/sam_altman/vhap_track"
TGT_FOLDER="$PROJECT_ROOT/data/sam_altman/transformsVHAP"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG_PATH"
}

log "watcher started"
while tmux has-session -t "$TRACK_SESSION" 2>/dev/null; do
    log "waiting for $TRACK_SESSION"
    sleep 60
done
log "$TRACK_SESSION session ended"

latest="$(find "$SRC_FOLDER" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
if [[ -z "$latest" ]]; then
    log "no tracking run directory found under $SRC_FOLDER; skipping export"
    exit 1
fi

if ! compgen -G "$latest/tracked_flame_params*.npz" > /dev/null; then
    log "no tracked_flame_params*.npz found in $latest; skipping export"
    exit 1
fi

log "exporting from $latest to $TGT_FOLDER"
cd "$PROJECT_ROOT/VHAP"
CUDA_VISIBLE_DEVICES=0 \
MPLCONFIGDIR=/tmp/matplotlib-vhap \
PYTHONPATH="$PROJECT_ROOT/VHAP" \
"$PROJECT_ROOT/.venv/bin/python" -u -m vhap.export_as_nerf_dataset \
    --src-folder "$SRC_FOLDER" \
    --tgt-folder "$TGT_FOLDER" \
    --background-color white >> "$LOG_PATH" 2>&1

log "export finished"
