"""
Meeting audio capture with rolling-window queue output.
"""

from __future__ import annotations

import threading
import time
from queue import Empty, Full, Queue

import numpy as np


class MeetingAudioListener:
    """
    Captures 16 kHz mono audio into a rolling window and pushes chunks periodically.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_seconds: int = 5,
        push_interval: int = 2,
        queue_maxsize: int = 10,
        input_device: int | str | None = None,
    ):
        self.sr = sample_rate
        self.chunk_seconds = chunk_seconds
        self.push_interval = push_interval
        self.chunk_size = sample_rate * chunk_seconds
        self.block_size = sample_rate * push_interval
        self.audio_queue: Queue[np.ndarray] = Queue(maxsize=queue_maxsize)
        self.input_device = input_device

        self._buffer = np.zeros(self.chunk_size, dtype=np.float32)
        self._buffer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pusher_thread: threading.Thread | None = None
        self._stream = None

    def _callback(self, indata, _frames, _time_info, _status):
        audio = indata[:, 0].astype(np.float32, copy=True)
        with self._buffer_lock:
            self._buffer = np.concatenate([self._buffer[len(audio) :], audio])

    def _push_loop(self):
        while not self._stop_event.is_set():
            with self._buffer_lock:
                chunk = self._buffer.copy()
            self._put_chunk(chunk)
            time.sleep(self.push_interval)

    def _put_chunk(self, chunk: np.ndarray):
        try:
            self.audio_queue.put_nowait(chunk)
        except Full:
            try:
                self.audio_queue.get_nowait()
            except Empty:
                pass
            self.audio_queue.put_nowait(chunk)

    def start(self):
        import sounddevice as sd

        if self._stream is not None:
            return
        self._stop_event.clear()
        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=self.block_size,
            device=self.input_device,
        )
        self._stream.start()
        self._pusher_thread = threading.Thread(target=self._push_loop, daemon=True)
        self._pusher_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._pusher_thread is not None:
            self._pusher_thread.join(timeout=1.0)
            self._pusher_thread = None
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

