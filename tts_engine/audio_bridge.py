"""
Bridge TTS output chunks into audio-driver-sized chunks.
"""

from __future__ import annotations

import threading
import time
from queue import Empty, Full, Queue
from typing import Callable

import numpy as np


class TTSAudioBridge:
    """
    Consumes TTS chunks, accumulates to HuBERT chunk size, and forwards chunks.
    """

    def __init__(
        self,
        audio_consumer: Callable[[np.ndarray], None],
        hubert_chunk_size: int = 3200,
        sample_rate: int = 16000,
        queue_maxsize: int = 50,
        realtime: bool = True,
    ):
        self.audio_consumer = audio_consumer
        self.hubert_chunk_size = int(hubert_chunk_size)
        self.sample_rate = int(sample_rate)
        self.realtime = bool(realtime)
        self.tts_audio_queue: Queue[np.ndarray] = Queue(maxsize=queue_maxsize)
        self._accumulator = np.array([], dtype=np.float32)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_emit_time: float | None = None

    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._next_emit_time = None
        self._thread = threading.Thread(target=self.feed_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._next_emit_time = None

    def put_chunk(self, chunk: np.ndarray):
        try:
            self.tts_audio_queue.put_nowait(chunk)
        except Full:
            try:
                self.tts_audio_queue.get_nowait()
            except Empty:
                pass
            self.tts_audio_queue.put_nowait(chunk)

    def feed_loop(self):
        while not self._stop_event.is_set():
            try:
                chunk = self.tts_audio_queue.get(timeout=0.2)
            except Empty:
                continue

            self._accumulator = np.concatenate([self._accumulator, chunk.astype(np.float32)])
            while len(self._accumulator) >= self.hubert_chunk_size:
                hubert_input = self._accumulator[: self.hubert_chunk_size].copy()
                self._accumulator = self._accumulator[self.hubert_chunk_size :]
                if not self._wait_until_next_emit():
                    return
                try:
                    self.audio_consumer(hubert_input)
                except Exception as exc:
                    print(f"[TTSAudioBridge] audio_consumer failed: {exc}")

    def _wait_until_next_emit(self) -> bool:
        if not self.realtime:
            return not self._stop_event.is_set()

        now = time.monotonic()
        chunk_duration = self.hubert_chunk_size / max(self.sample_rate, 1)
        if self._next_emit_time is None or now > self._next_emit_time + 1.0:
            self._next_emit_time = now

        delay = self._next_emit_time - now
        if delay > 0 and self._stop_event.wait(delay):
            return False

        self._next_emit_time += chunk_duration
        return not self._stop_event.is_set()
