# Scripts

Utility scripts for training, evaluation, and data processing.

## Scripts
- `train_audio_driver.py` — Train the speaker-specific MotionTranslator
- `preprocess_data.py` — Validate/prepare subject data + generate persona template
- `prepare_avatar_training_source.py` — Build a clean master/pilot video from a trim manifest
- `run_avatar_retrain_pipeline.sh` — Run VHAP export, GaussianAvatars training, and audio-driver training for a new subject
- `vhap_progress.py` — Read-only VHAP tracking progress/ETA monitor
- `audio_progress.py` — Read-only MotionTranslator audio-driver progress/ETA monitor
- `prepare_demo_character.py` — Extract and wire a downloaded demo character into `data/{subject}`
- `setup_v4l2loopback.sh` — Set up virtual camera on Fedora (`OBS_EXCLUSIVE_CAPS` / `AVATAR_EXCLUSIVE_CAPS` overridable)

## VHAP tracking monitor

```bash
python scripts/vhap_progress.py --subject new_speaker --mode pilot --watch
```

For the full retraining pass, switch `--mode full`. The monitor only reads logs and GPU status.

## Audio-driver training monitor

```bash
python scripts/audio_progress.py --subject new_speaker --mode pilot --watch
```

For the full audio-driver pass, switch `--mode full`. The monitor only reads the audio training log, checkpoint mtimes, tmux status, and GPU status.
