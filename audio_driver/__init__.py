"""
Audio Driver Package
====================
Converts live 16kHz audio into FLAME expression + jaw parameters.

Components:
  - audio_encoder.py   — HuBERT feature extraction (frozen)
  - motion_translator.py — Transformer mapping audio → FLAME params
  - inference.py       — StreamingAudioDriver: real-time end-to-end pipeline
"""

from .audio_encoder import AudioEncoder
from .motion_translator import MotionTranslator
from .inference import StreamingAudioDriver

__all__ = ["AudioEncoder", "MotionTranslator", "StreamingAudioDriver"]
