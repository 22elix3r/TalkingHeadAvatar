#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_avatar_retrain_pipeline.sh --subject SUBJECT --mode pilot|full --step STEP [options]

Steps:
  preprocess       Extract frames and alpha maps with VHAP preprocessing.
  track            Run VHAP FLAME tracking.
  export           Export latest VHAP tracking run to transformsVHAP.
  wire             Create data/{subject}/view_000 image/mask symlinks.
  preprocess-data  Validate subject data and extract helper audio/persona files.
  ga               Train GaussianAvatars.
  audio            Train the subject-specific MotionTranslator audio driver.
  validate         Run local dataset validation.
  all-data         preprocess -> track -> export -> wire -> preprocess-data -> validate.
  all              all-data -> ga -> audio.

Options:
  --subject NAME       Subject slug. Default: new_speaker
  --mode MODE          pilot or full. Default: pilot
  --step STEP          Pipeline step. Default: validate
  --fps N             VHAP preprocessing FPS. Default: 25
  --ga-iterations N   GaussianAvatars iterations. Default: 60000 for pilot, 300000 for full
  --ga-interval N     Save/eval/checkpoint interval. Default: 20000 for pilot, 50000 for full
  --ga-port N         GaussianAvatars GUI port. Default: 60000
  --ga-resume PATH    Resume GaussianAvatars from chkpnt*.pth. Use "latest" or an iteration number.
  --vhap-batch N      VHAP tracking batch size. Default: 8
  --vhap-resume PATH  Resume VHAP tracking from a checkpoint .npz. Use "latest" for newest checkpoint.
  --audio-epochs N    Audio driver epochs. Default: 80 for pilot, 150 for full
  --audio-batch N     Audio driver batch size. Default: 32
  --audio-no-resume   Start audio-driver training from scratch.

Run long stages inside tmux from the project root, for example:
  tmux new-session -d -s avatar_pilot "bash -lc 'cd /home/elix3r/projects/TalkingHeadAvatar && source activate_env.sh && scripts/run_avatar_retrain_pipeline.sh --subject new_speaker --mode pilot --step all-data'"
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBJECT="new_speaker"
MODE="pilot"
STEP="validate"
FPS="25"
GA_ITERATIONS=""
GA_INTERVAL=""
GA_PORT="60000"
GA_RESUME=""
VHAP_BATCH="8"
VHAP_RESUME=""
AUDIO_EPOCHS=""
AUDIO_BATCH="32"
AUDIO_NO_RESUME="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject) SUBJECT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --ga-iterations) GA_ITERATIONS="$2"; shift 2 ;;
    --ga-interval) GA_INTERVAL="$2"; shift 2 ;;
    --ga-port) GA_PORT="$2"; shift 2 ;;
    --ga-resume) GA_RESUME="$2"; shift 2 ;;
    --vhap-batch) VHAP_BATCH="$2"; shift 2 ;;
    --vhap-resume) VHAP_RESUME="$2"; shift 2 ;;
    --audio-epochs) AUDIO_EPOCHS="$2"; shift 2 ;;
    --audio-batch) AUDIO_BATCH="$2"; shift 2 ;;
    --audio-no-resume) AUDIO_NO_RESUME="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$MODE" != "pilot" && "$MODE" != "full" ]]; then
  echo "--mode must be 'pilot' or 'full'" >&2
  exit 2
fi

if [[ "$MODE" == "pilot" ]]; then
  : "${GA_ITERATIONS:=60000}"
  : "${GA_INTERVAL:=20000}"
  : "${AUDIO_EPOCHS:=80}"
  VIDEO_NAME="pilot_video.mp4"
  TRANSFORMS_DIR="$PROJECT_ROOT/data/$SUBJECT/transformsVHAP_pilot"
  GA_MODEL_DIR="$PROJECT_ROOT/output/$SUBJECT/ga_pilot"
else
  : "${GA_ITERATIONS:=300000}"
  : "${GA_INTERVAL:=50000}"
  : "${AUDIO_EPOCHS:=150}"
  VIDEO_NAME="master_video.mp4"
  TRANSFORMS_DIR="$PROJECT_ROOT/data/$SUBJECT/transformsVHAP"
  GA_MODEL_DIR="$PROJECT_ROOT/output/$SUBJECT/ga_phase4_enhanced"
fi

SOURCE_VIDEO="$PROJECT_ROOT/data/$SUBJECT/master/$VIDEO_NAME"
MONOCULAR_ROOT="$PROJECT_ROOT/data/monocular"
SEQUENCE="${SUBJECT}_${MODE}"
MONOCULAR_VIDEO="$MONOCULAR_ROOT/$SEQUENCE.mp4"
TRACK_ROOT="$PROJECT_ROOT/output/$SUBJECT/vhap_track"
LOG_DIR="$PROJECT_ROOT/output/$SUBJECT/logs"
AUDIO_CKPT_DIR="$PROJECT_ROOT/audio_driver/checkpoints/$SUBJECT"
AUDIO_PATH="$PROJECT_ROOT/data/$SUBJECT/audio_full.wav"
if [[ "$MODE" == "pilot" ]]; then
  AUDIO_PATH="$PROJECT_ROOT/data/$SUBJECT/master/pilot_audio.wav"
fi
AUDIO_FEATURE_CACHE="$AUDIO_CKPT_DIR/hubert_features_${MODE}.pt"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/VHAP:${PYTHONPATH:-}"
if [[ -f "$PROJECT_ROOT/activate_env.sh" ]]; then
  # shellcheck source=/dev/null
  set +u
  source "$PROJECT_ROOT/activate_env.sh"
  set -u
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

latest_track_dir() {
  find "$TRACK_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

latest_vhap_checkpoint() {
  find "$TRACK_ROOT" -mindepth 2 -maxdepth 2 -type f -name 'checkpoint_flame_params*.npz' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

latest_ga_checkpoint() {
  find "$GA_MODEL_DIR" -maxdepth 1 -type f -name 'chkpnt*.pth' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

checkpoint_processed_frames() {
  python -c 'import sys, numpy as np; z=np.load(sys.argv[1]); print(int(z["n_processed_frames"]))' "$1"
}

resolve_vhap_resume_checkpoint() {
  local checkpoint="$VHAP_RESUME"
  if [[ -z "$checkpoint" ]]; then
    return 0
  fi
  if [[ "$checkpoint" == "latest" ]]; then
    checkpoint="$(latest_vhap_checkpoint)"
  elif [[ "$checkpoint" != /* ]]; then
    checkpoint="$PROJECT_ROOT/$checkpoint"
  fi
  if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
    echo "No VHAP checkpoint found for --vhap-resume '$VHAP_RESUME'" >&2
    exit 1
  fi
  printf '%s\n' "$checkpoint"
}

resolve_ga_resume_checkpoint() {
  local checkpoint="$GA_RESUME"
  if [[ -z "$checkpoint" ]]; then
    return 0
  fi
  if [[ "$checkpoint" == "latest" ]]; then
    checkpoint="$(latest_ga_checkpoint)"
  elif [[ "$checkpoint" =~ ^[0-9]+$ ]]; then
    checkpoint="$GA_MODEL_DIR/chkpnt${checkpoint}.pth"
  elif [[ "$checkpoint" != /* ]]; then
    checkpoint="$PROJECT_ROOT/$checkpoint"
  fi
  if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
    echo "No GaussianAvatars checkpoint found for --ga-resume '$GA_RESUME'" >&2
    exit 1
  fi
  printf '%s\n' "$checkpoint"
}

link_monocular_video() {
  require_file "$SOURCE_VIDEO"
  mkdir -p "$MONOCULAR_ROOT" "$TRACK_ROOT" "$LOG_DIR"
  ln -sfn "$SOURCE_VIDEO" "$MONOCULAR_VIDEO"
  echo "[pipeline] Monocular source: $MONOCULAR_VIDEO -> $SOURCE_VIDEO"
}

step_preprocess() {
  link_monocular_video
  (
    cd "$PROJECT_ROOT/VHAP"
    python vhap/preprocess_video.py \
      --input "$MONOCULAR_VIDEO" \
      --target_fps "$FPS" \
      --matting_method robust_video_matting
  ) 2>&1 | tee -a "$LOG_DIR/vhap_preprocess_${MODE}.log"
}

step_track() {
  link_monocular_video
  local resume_checkpoint=""
  local resume_timestep=""
  local resume_args=()
  resume_checkpoint="$(resolve_vhap_resume_checkpoint)"
  if [[ -n "$resume_checkpoint" ]]; then
    resume_timestep="$(checkpoint_processed_frames "$resume_checkpoint")"
    resume_args=(
      --model.flame-params-path "$resume_checkpoint"
      --begin-timestep "$resume_timestep"
    )
    echo "[pipeline] Resuming VHAP from: $resume_checkpoint"
    echo "[pipeline] Resuming at timestep: $resume_timestep"
  fi
  (
    cd "$PROJECT_ROOT/VHAP"
    python vhap/track.py \
      --data.root_folder "$MONOCULAR_ROOT" \
      --exp.output_folder "$TRACK_ROOT" \
      --data.sequence "$SEQUENCE" \
      --batch-size "$VHAP_BATCH" \
      --no-async-func \
      "${resume_args[@]}"
  ) 2>&1 | tee -a "$LOG_DIR/vhap_track_${MODE}.log"
}

step_export() {
  local src
  src="$(latest_track_dir)"
  if [[ -z "$src" ]]; then
    echo "No VHAP tracking directory found under $TRACK_ROOT" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$TRANSFORMS_DIR")"
  (
    cd "$PROJECT_ROOT/VHAP"
    python vhap/export_as_nerf_dataset.py \
      --src_folder "$src" \
      --tgt_folder "$TRANSFORMS_DIR" \
      --background-color white
  ) 2>&1 | tee -a "$LOG_DIR/vhap_export_${MODE}.log"
}

step_wire() {
  mkdir -p "$PROJECT_ROOT/data/$SUBJECT/view_000"
  ln -sfn "../$(basename "$TRANSFORMS_DIR")/images" "$PROJECT_ROOT/data/$SUBJECT/view_000/images"
  ln -sfn "../$(basename "$TRANSFORMS_DIR")/fg_masks" "$PROJECT_ROOT/data/$SUBJECT/view_000/masks"
  echo "[pipeline] view_000/images -> ../$(basename "$TRANSFORMS_DIR")/images"
  echo "[pipeline] view_000/masks  -> ../$(basename "$TRANSFORMS_DIR")/fg_masks"
}

step_preprocess_data() {
  local video_for_audio="$PROJECT_ROOT/data/$SUBJECT/master/master_video.mp4"
  if [[ "$MODE" == "pilot" && ! -f "$video_for_audio" ]]; then
    video_for_audio="$SOURCE_VIDEO"
  fi
  require_file "$video_for_audio"
  python scripts/preprocess_data.py \
    --subject "$SUBJECT" \
    --video "$video_for_audio" \
    2>&1 | tee -a "$LOG_DIR/preprocess_data_${MODE}.log"
}

step_validate() {
  python scripts/prepare_avatar_training_source.py --subject "$SUBJECT" --validate || true
  python scripts/preprocess_data.py --subject "$SUBJECT"
}

step_ga() {
  mkdir -p "$LOG_DIR"
  local resume_checkpoint=""
  local resume_args=()
  resume_checkpoint="$(resolve_ga_resume_checkpoint)"
  if [[ -n "$resume_checkpoint" ]]; then
    resume_args=(--start_checkpoint "$resume_checkpoint")
    echo "[pipeline] Resuming GaussianAvatars from: $resume_checkpoint"
  fi
  python GaussianAvatars/train.py \
    -s "$TRANSFORMS_DIR" \
    -m "$GA_MODEL_DIR" \
    --eval \
    --bind_to_mesh \
    --white_background \
    --iterations "$GA_ITERATIONS" \
    --enable_aps \
    --enable_mouth_structure \
    --enable_partwise_deformation \
    --lambda_rigid 10.0 \
    --lambda_flex 0.1 \
    --interval "$GA_INTERVAL" \
    --port "$GA_PORT" \
    --data_loader_workers 0 \
    "${resume_args[@]}" \
    2>&1 | tee -a "$LOG_DIR/ga_${MODE}.log"
}

step_audio() {
  mkdir -p "$AUDIO_CKPT_DIR" "$LOG_DIR"
  require_file "$AUDIO_PATH"
  local resume_args=()
  if [[ "$AUDIO_NO_RESUME" == "1" ]]; then
    resume_args=(--no_resume)
  fi
  python scripts/train_audio_driver.py \
    --flame_params "$TRANSFORMS_DIR/flame_param" \
    --audio_path "$AUDIO_PATH" \
    --output_dir "$AUDIO_CKPT_DIR" \
    --transforms_dir "$TRANSFORMS_DIR" \
    --epochs "$AUDIO_EPOCHS" \
    --lr 1e-4 \
    --batch_size "$AUDIO_BATCH" \
    --context_frames 50 \
    --feature_cache "$AUDIO_FEATURE_CACHE" \
    --num_workers 0 \
    "${resume_args[@]}" \
    2>&1 | tee -a "$LOG_DIR/audio_driver_${MODE}.log"
}

case "$STEP" in
  preprocess) step_preprocess ;;
  track) step_track ;;
  export) step_export ;;
  wire) step_wire ;;
  preprocess-data) step_preprocess_data ;;
  validate) step_validate ;;
  ga) step_ga ;;
  audio) step_audio ;;
  all-data)
    step_preprocess
    step_track
    step_export
    step_wire
    step_preprocess_data
    step_validate
    ;;
  all)
    step_preprocess
    step_track
    step_export
    step_wire
    step_preprocess_data
    step_validate
    step_ga
    step_audio
    ;;
  *) echo "Unknown --step: $STEP" >&2; usage; exit 2 ;;
esac
