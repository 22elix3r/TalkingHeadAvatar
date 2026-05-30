"""
Abstract TTS interface for swappable implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from queue import Queue

import numpy as np


class BaseTTSEngine(ABC):
    @abstractmethod
    def load_voice(self, reference_path: str) -> None:
        """Load and cache voice embedding from a reference clip."""

    @abstractmethod
    def synthesize_streaming(
        self,
        text: str,
        audio_queue: Queue[np.ndarray],
        emotion_mode: str = "neutral",
    ) -> None:
        """Push 16 kHz mono float32 chunks to ``audio_queue``."""
