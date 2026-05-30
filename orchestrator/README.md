# Orchestrator Module (Gemma 4 E4B)

This module handles the conversational AI orchestration layer using Gemma 4 E4B with 4-bit quantization.

## Components
- `gemma_loader.py` — Model loading with NF4 quantization + Flash Attention 2
- `audio_listener.py` — Meeting audio capture via sounddevice (16kHz rolling windows)
- `stt.py` — Whisper-based speech-to-text over rolling audio windows
- `persona.py` — Persona injection and system prompt construction
- `transcript.py` — Rolling meeting transcript with 128K context management

## Data Flow
```
Meeting Audio → STT (Whisper) → Gemma 4 E4B (with persona) → avatar_speak function call → TTS queue
```

## See Also
- Phase 7 skill: `.agent/skills/phase7_gemma_orchestration/skill.md`
- Implementation plan: `implementation_plan.md` (Phase 7)
