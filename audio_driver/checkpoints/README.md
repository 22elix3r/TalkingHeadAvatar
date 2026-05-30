# Audio Driver Checkpoints

This directory is intentionally tracked only through this README. Local training checkpoints and cached features are ignored because they are large binary artifacts.

Expected local files:

```text
audio_driver/checkpoints/{subject_or_run}/
  best_model.pt
  last_model.pt
  hubert_features_raw.pt
```

See the root [ARTIFACTS.md](../../ARTIFACTS.md) for the current local inventory.
