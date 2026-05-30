"""
Chatterbox-based streaming synthesizer.
"""

from __future__ import annotations

import inspect
import re
from queue import Empty, Full, Queue

import numpy as np
import torch
import torchaudio

from .base import BaseTTSEngine


class ChatterboxEngine(BaseTTSEngine):
    """
    Streaming TTS wrapper for Chatterbox-Turbo.
    """

    def __init__(
        self,
        voice_ref_path: str | None = None,
        device: str = "cuda",
        source_sample_rate: int = 24000,
        target_sample_rate: int = 16000,
        chunk_size: int = 4800,
    ):
        self.device = device
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.chunk_size = chunk_size
        self.voice_ref_path = voice_ref_path
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        from chatterbox.tts import ChatterboxTTS

        self._model = ChatterboxTTS.from_pretrained(device=self.device)
        if self.voice_ref_path:
            self.load_voice(self.voice_ref_path)

    def load_voice(self, reference_path: str) -> None:
        self._ensure_model()
        self.voice_ref_path = reference_path
        if hasattr(self._model, "load_speaker"):
            self._model.load_speaker(reference_path)
            return
        if hasattr(self._model, "prepare_conditionals"):
            self._model.prepare_conditionals(reference_path)
            return
        raise RuntimeError(
            "Installed ChatterboxTTS does not expose load_speaker() or "
            "prepare_conditionals(); cannot load voice reference."
        )

    def synthesize_streaming(
        self,
        text: str,
        audio_queue: Queue[np.ndarray],
        emotion_mode: str = "neutral",
    ) -> None:
        self._ensure_model()
        text = text.strip()
        if not text:
            return

        exaggeration = self._emotion_to_exaggeration(emotion_mode)
        if hasattr(self._model, "generate_stream"):
            stream = self._call_generate_stream(text=text, exaggeration=exaggeration)
            for chunk in stream:
                self._put_chunk(audio_queue, self._resample_to_16k(self._to_np(chunk)))
            return

        self.synthesize_sentence_by_sentence(
            text=text,
            audio_queue=audio_queue,
            emotion_mode=emotion_mode,
        )

    def synthesize_sentence_by_sentence(
        self,
        text: str,
        audio_queue: Queue[np.ndarray],
        emotion_mode: str = "neutral",
    ) -> None:
        self._ensure_model()
        exaggeration = self._emotion_to_exaggeration(emotion_mode)
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for sentence in sentences:
            if not sentence:
                continue
            audio = self._call_generate(sentence, exaggeration=exaggeration)
            chunk = self._resample_to_16k(self._to_np(audio))
            self._put_chunk(audio_queue, chunk)

    def _resample_to_16k(self, audio: np.ndarray) -> np.ndarray:
        waveform = torch.from_numpy(audio).float().view(1, -1)
        resampled = torchaudio.functional.resample(
            waveform,
            self.source_sample_rate,
            self.target_sample_rate,
        )
        return resampled.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _to_np(chunk) -> np.ndarray:
        if isinstance(chunk, np.ndarray):
            return chunk.astype(np.float32, copy=False)
        if torch.is_tensor(chunk):
            return chunk.detach().cpu().numpy().astype(np.float32, copy=False)
        return np.asarray(chunk, dtype=np.float32)

    @staticmethod
    def _put_chunk(audio_queue: Queue[np.ndarray], chunk: np.ndarray) -> None:
        try:
            audio_queue.put_nowait(chunk)
        except Full:
            try:
                audio_queue.get_nowait()
            except Empty:
                pass
            audio_queue.put_nowait(chunk)

    def _call_generate_stream(self, text: str, exaggeration: float):
        kwargs = {"text": text, "chunk_size": self.chunk_size}
        self._inject_if_supported(self._model.generate_stream, kwargs, "exaggeration", exaggeration)
        return self._model.generate_stream(**kwargs)

    def _call_generate(self, text: str, exaggeration: float):
        kwargs = {"text": text}
        self._inject_if_supported(self._model.generate, kwargs, "exaggeration", exaggeration)
        return self._model.generate(**kwargs)

    @staticmethod
    def _inject_if_supported(fn, kwargs: dict, name: str, value):
        signature = inspect.signature(fn)
        if name in signature.parameters:
            kwargs[name] = value

    @staticmethod
    def _emotion_to_exaggeration(emotion_mode: str) -> float:
        mapping = {
            "neutral": 0.45,
            "engaged": 0.65,
            "emphatic": 0.9,
            "concerned": 0.6,
        }
        return mapping.get(emotion_mode, mapping["neutral"])
