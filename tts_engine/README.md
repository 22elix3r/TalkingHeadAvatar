# TTS Engine Module (Chatterbox-Turbo)

This module provides streaming text-to-speech synthesis with zero-shot voice cloning.

## Components
- `base.py` — Abstract `BaseTTSEngine` interface (swappable implementations)
- `synthesizer.py` — Chatterbox-Turbo streaming synthesis (200ms chunks, 24kHz → 16kHz resample)
- `audio_bridge.py` — Bridge between TTS output and HuBERT audio driver

## Data Flow
```
Text from Gemma → Chatterbox TTS (streaming) → Resample 24→16kHz → Audio Bridge → HuBERT → Avatar
```

## VRAM: ~1.5–2.0 GB
## First-chunk latency: ~75ms

## See Also
- Phase 8 skill: `.agent/skills/phase8_tts_integration/skill.md`
- Implementation plan: `implementation_plan.md` (Phase 8)
