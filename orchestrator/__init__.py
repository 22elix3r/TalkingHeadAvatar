"""
orchestrator — Gemma 4 E4B multimodal orchestration layer.

Components:
  gemma_loader   — load Gemma with 4-bit quantisation
  audio_listener — streaming meeting audio capture
  stt            — speech-to-text transcription over meeting audio
  persona        — persona injection + system prompt
  transcript     — rolling meeting transcript
"""
from .gemma_loader import load_gemma, GEMMA_MODEL_ID
from .gemma_loader import (
    AVATAR_TOOLS,
    build_audio_generation_inputs,
    process_gemma_output,
    split_first_sentence,
    stream_gemma_with_early_tts,
)
from .audio_listener import MeetingAudioListener
from .persona import build_system_prompt, load_persona
from .stt import MeetingSpeechRecognizer
from .transcript import MeetingTranscript

__all__ = [
    "load_gemma",
    "GEMMA_MODEL_ID",
    "AVATAR_TOOLS",
    "build_audio_generation_inputs",
    "process_gemma_output",
    "split_first_sentence",
    "stream_gemma_with_early_tts",
    "MeetingAudioListener",
    "MeetingSpeechRecognizer",
    "build_system_prompt",
    "load_persona",
    "MeetingTranscript",
]
