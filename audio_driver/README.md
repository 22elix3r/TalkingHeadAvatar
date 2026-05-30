# Audio Driver Module

This module converts live audio into FLAME expression parameters using HuBERT-based audio feature extraction and a speaker-specific Motion Translator.

## Components
- `audio_encoder.py` — HuBERT feature extraction (frozen weights)
- `motion_translator.py` — Transformer-based audio-to-FLAME mapping
- `checkpoints/` — Per-subject trained Motion Translator weights

## Data Flow
```
Audio waveform (16kHz) → HuBERT → features (B, T', 1024) → MotionTranslator → expression (50) + jaw_pose (3)
```

## See Also
- Phase 5 skill: `.agent/skills/phase5_audio_driver/skill.md`
- Implementation plan: `implementation_plan.md` (Phase 5)
