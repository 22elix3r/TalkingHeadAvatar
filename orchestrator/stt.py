"""
Speech-to-text helpers for live meeting audio.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch


def normalize_transcript_text(text: str) -> str:
    """
    Normalize transcript text for deduplication and overlap checks.
    """
    return " ".join(
        "".join(ch for ch in token.lower() if ch.isalnum())
        for token in text.split()
        if "".join(ch for ch in token.lower() if ch.isalnum())
    )


def _suffix_prefix_overlap(previous: list[str], current: list[str], min_words: int) -> int:
    limit = min(len(previous), len(current))
    for size in range(limit, min_words - 1, -1):
        if previous[-size:] == current[:size]:
            return size
    return 0


@dataclass(slots=True)
class RollingTextDeduplicator:
    """
    Suppresses repeated ASR fragments caused by overlapping audio windows.
    """

    min_overlap_words: int = 3
    _last_normalized: str = ""

    def filter(self, text: str) -> str:
        candidate = text.strip()
        if not candidate:
            return ""

        normalized = normalize_transcript_text(candidate)
        if not normalized:
            return ""

        if normalized == self._last_normalized:
            return ""
        if normalized in self._last_normalized:
            return ""

        previous_tokens = self._last_normalized.split()
        current_tokens = normalized.split()
        overlap = _suffix_prefix_overlap(
            previous=previous_tokens,
            current=current_tokens,
            min_words=self.min_overlap_words,
        )
        if overlap > 0:
            original_words = candidate.split()
            if overlap >= len(original_words):
                self._last_normalized = normalized
                return ""
            candidate = " ".join(original_words[overlap:]).strip()
            normalized = normalize_transcript_text(candidate)
            if not normalized:
                return ""

        self._last_normalized = normalized
        return candidate

    def reset(self):
        self._last_normalized = ""


class MeetingSpeechRecognizer:
    """
    Whisper-based ASR wrapper using the transformers pipeline.
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-tiny.en",
        device: str = "cuda",
        language: str = "en",
        min_audio_rms: float = 0.003,
    ):
        self.model_id = model_id
        self.device = device
        self.language = language
        self.min_audio_rms = min_audio_rms
        self._pipeline = None
        self._pipeline_lock = threading.Lock()
        self._deduplicator = RollingTextDeduplicator()

    def _resolve_pipeline_device(self) -> int:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            if ":" in self.device:
                return int(self.device.split(":", maxsplit=1)[1])
            return 0
        return -1

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        with self._pipeline_lock:
            if self._pipeline is not None:
                return self._pipeline
            from transformers import pipeline

            pipeline_device = self._resolve_pipeline_device()
            torch_dtype = torch.float16 if pipeline_device >= 0 else torch.float32
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=pipeline_device,
                torch_dtype=torch_dtype,
            )
            return self._pipeline

    def transcribe_chunk(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe one audio chunk and return a deduplicated text fragment.
        """
        chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return ""

        rms = float(np.sqrt(np.mean(np.square(chunk))))
        if rms < self.min_audio_rms:
            return ""

        asr_pipeline = self._ensure_pipeline()
        generate_kwargs = {"task": "transcribe"}
        if self.language:
            generate_kwargs["language"] = self.language

        result = asr_pipeline(
            {"array": chunk, "sampling_rate": sample_rate},
            generate_kwargs=generate_kwargs,
        )
        text = str(result.get("text", "")).strip()
        if not text:
            return ""
        return self._deduplicator.filter(text)

    def warmup(self):
        """
        Eagerly load the ASR pipeline to fail fast during startup.
        """
        self._ensure_pipeline()

    def reset(self):
        self._deduplicator.reset()
