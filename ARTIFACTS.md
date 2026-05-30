# Local Artifact Inventory

This file tracks references to large local artifacts that are intentionally excluded from git. It exists so the workspace is understandable without pushing generated binaries, datasets, checkpoints, or model weights into GitHub.

Generated on: 2026-05-30

## Summary

| Path | Approx Size | Tracked In Git | Notes |
|---|---:|---|---|
| `data/` | 5.2 GB | README only | Subject data, demo downloads, videos, audio, VHAP/Metracker transforms. |
| `output/` | 51 GB | README only | Training outputs, checkpoints, TensorBoard events, exported tarballs, rendered assets. |
| `audio_driver/checkpoints/` | 582 MB | README only | Audio driver checkpoints and cached HuBERT features. |
| `models/` | 0 B | README only | Intended location for downloaded LLM/TTS/STT/model weights. |
| `.venv/` | 8.5 GB | ignored | Local Python environment. |
| `.git-local/` | 9.6 MB | ignored | Local git database used in this environment. |

## Largest Local Artifacts

These are examples of files intentionally kept out of git:

| Size | Path |
|---:|---|
| 532 MB | `output/sam_altman/ga_phase4_recovered_300k/point_cloud/iteration_300000/flame_param.npz` |
| 532 MB | `output/sam_altman/ga_phase4_enhanced/point_cloud/iteration_300000/flame_param.npz` |
| 528 MB | `data/demo_character/splattingavatar.tarkkrpepv2.part` |
| 440 MB | `data/demo_character/splattingavatar.tar3beq0ray.part` |
| 405 MB | `audio_driver/checkpoints/sam_altman_phase5/last_model.pt` |
| 303 MB | `output/demo_character/splattingavatar/marcel_001.tar` |
| 272 MB | `output/demo_character/splattingavatar/bala_001.tar` |
| 234 MB | `output/demo_character/splattingavatar/wojtek_001.tar` |
| 226 MB | `output/demo_character/splattingavatar/malte_001.tar` |
| 223 MB | `output/demo_character/splattingavatar/nf_001.tar` |
| 222 MB | `output/demo_character/splattingavatar/nf_003.tar` |
| 203 MB | `output/demo_character/dynamicface/chkim_001.tar` |
| 190 MB | `output/demo_character/splattingavatar/biden_001.tar` |
| 187 MB | `output/demo_character/dynamicface/hmlew_001.tar` |
| 181 MB | `output/demo_character/splattingavatar/obama_001.tar` |
| 179 MB | `output/demo_character/dynamicface/dycho_001.tar` |
| 176 MB | `output/demo_character/extracted/biden_001/point_cloud/iteration_180000/flame_param.npz` |
| 175 MB | `output/demo_character/splattingavatar/yufeng_001.tar` |
| 170 MB | `output/demo_character/dynamicface/jskang_001.tar` |
| 167 MB | `output/demo_character/dynamicface/sjmoon_001.tar` |
| 135 MB | `audio_driver/checkpoints/sam_altman_phase5/best_model.pt` |
| 102 MB | `output/demo_character/splattingavatar/person_004.tar` |
| 95 MB | `output/sam_altman/ga_phase4_enhanced/events.out.tfevents.1779994163.fedora.400092.0` |
| 95 MB | `output/sam_altman/ga_phase4_enhanced/events.out.tfevents.1780003090.fedora.41165.0` |
| 89 MB | `output/sam_altman/ga_phase4_recovered_300k/chkpnt300000.pth` |
| 87 MB | `output/sam_altman/ga_phase4_enhanced/chkpnt300000.pth` |
| 80 MB | `data/sam_altman/video.mp4` |
| 70 MB | `audio_driver/checkpoints/sam_altman_phase5/hubert_features_raw.pt` |
| 70 MB | `GaussianAvatars/media/306/flame_param.npz` |

## Directory Contracts

`data/`

Expected to contain subject-level training inputs:

```text
data/{subject}/
  transformsVHAP/
  view_000/
  audio_full.wav
  voice_reference.wav
  persona.json
```

`output/`

Expected to contain generated training and render outputs:

```text
output/{subject_or_run}/
  point_cloud/
  checkpoints
  TensorBoard event files
  exported splats or archives
```

`audio_driver/checkpoints/`

Expected to contain per-subject audio-to-FLAME weights:

```text
audio_driver/checkpoints/{subject_or_run}/
  best_model.pt
  last_model.pt
  hubert_features_raw.pt
```

`models/`

Expected to contain downloaded runtime models:

```text
models/
  *.gguf
  *.safetensors
  other local model weights
```

## Moving Artifacts Later

If these artifacts need to be shared between machines, use one of these rather than normal git blobs:

- GitHub Releases for a small curated demo package.
- Git LFS for selected files, after checking quota and cost.
- Cloud/object storage for full datasets and training outputs.
- A dataset registry such as Hugging Face Datasets for public/shareable data.
