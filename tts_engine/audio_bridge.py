"""
Bridge TTS output chunks into audio-driver-sized chunks.
"""

from __future__ import annotations

import threading
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
        queue_maxsize: int = 50,
    ):
        self.audio_consumer = audio_consumer
        self.hubert_chunk_size = hubert_chunk_size
        self.tts_audio_queue: Queue[np.ndarray] = Queue(maxsize=queue_maxsize)
        self._accumulator = np.array([], dtype=np.float32)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.feed_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

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
                hubert_input = self._accumulator[: self.hubert_chunk_size]
                self._accumulator = self._accumulator[self.hubert_chunk_size :]
                self.audio_consumer(hubert_input)

