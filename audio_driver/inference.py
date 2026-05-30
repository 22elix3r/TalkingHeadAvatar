"""
Streaming Audio Driver — Real-Time Inference Pipeline
=====================================================
Connects audio input → HuBERT encoder → MotionTranslator → FLAME params.
Designed for <10 ms per chunk latency when used with the rendering loop.

Usage:
    driver = StreamingAudioDriver(
        motion_translator_ckpt="audio_driver/checkpoints/subject/model.pt",
        device="cuda",
    )
    # In render loop:
    audio_chunk = ...  # (1, chunk_size) at 16 kHz
    expr, jaw = driver.step(audio_chunk)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .audio_encoder import AudioEncoder
from .motion_translator import MotionTranslator


class StreamingAudioBuffer:
    """
    Sliding window audio buffer for causal real-time inference.

    Args:
        chunk_size:   Number of samples per feature window (e.g. 16000 = 1 s).
        hop_size:     Samples consumed per inference step (e.g. 320 = 20 ms).
        sample_rate:  Audio sample rate (must match HuBERT: 16 kHz).
    """

    def __init__(
        self,
        chunk_size: int = 16000,   # 1 second context
        hop_size: int = 320,       # 1 HuBERT frame (20 ms)
        sample_rate: int = 16000,
    ):
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate

        self._buffer = torch.zeros(chunk_size, dtype=torch.float32)
        self._lock = threading.Lock()

    def push(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Push new audio samples into the buffer and return the current window.

        Args:
            samples: (N,) float32 tensor — new audio samples (N ≤ chunk_size).

        Returns:
            window: (1, chunk_size) ready for AudioEncoder.
        """
        n = samples.shape[0]
        with self._lock:
            self._buffer = torch.cat([self._buffer[n:], samples.cpu().float()])
            return self._buffer.clone().unsqueeze(0)  # (1, chunk_size)

    def reset(self):
        with self._lock:
            self._buffer = torch.zeros(self.chunk_size, dtype=torch.float32)


class StreamingAudioDriver:
    """
    End-to-end real-time audio → FLAME parameters pipeline.

    Maintains:
      - AudioEncoder (HuBERT, frozen, FP16)
      - MotionTranslator (subject-specific, loaded from checkpoint)
      - StreamingAudioBuffer (sliding window)
      - Exponential moving average for temporal smoothing

    Args:
        motion_translator_ckpt: Path to .pt checkpoint from train_audio_driver.py.
        device:                 'cuda' or 'cpu'.
        fp16:                   Use FP16 for HuBERT.
        ema_alpha:              Smoothing factor (0 = no smoothing, 1 = no update).
        context_seconds:        Audio context window in seconds (default: 1 s).
    """

    def __init__(
        self,
        motion_translator_ckpt: Optional[str | Path] = None,
        device: str | torch.device = "cuda",
        fp16: bool = True,
        ema_alpha: float = 0.3,
        context_seconds: float = 1.0,
    ):
        self.device = torch.device(device)
        self.ema_alpha = ema_alpha

        # Audio encoder (lazy-loaded HuBERT)
        self.encoder = AudioEncoder(device=device, fp16=fp16)

        # Motion translator
        if motion_translator_ckpt is not None:
            self.translator = MotionTranslator.load(motion_translator_ckpt, device=device)
            print(f"[StreamingAudioDriver] Loaded translator from {motion_translator_ckpt}")
        else:
            # Untrained translator — useful for testing pipeline structure
            self.translator = MotionTranslator(causal=True).to(device)
            print("[StreamingAudioDriver] WARNING: using untrained MotionTranslator")

        self.translator.eval()

        # Streaming buffer
        sr = 16000
        chunk_size = int(context_seconds * sr)
        self.buffer = StreamingAudioBuffer(chunk_size=chunk_size, hop_size=320, sample_rate=sr)

        # EMA state
        self._ema_expr: Optional[torch.Tensor] = None
        self._ema_jaw: Optional[torch.Tensor] = None

        # Latency tracking
        self._last_latency_ms: float = 0.0

    @torch.no_grad()
    def step(self, audio_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process a new audio chunk and return the latest FLAME parameters.

        Args:
            audio_chunk: (N,) or (1, N) float32 at 16 kHz.

        Returns:
            expression: (n_expr,) — FLAME expression coefficients for the current frame.
            jaw_pose:   (3,)      — jaw rotation axis-angle (radians).
        """
        t0 = time.perf_counter()

        if audio_chunk.dim() == 2:
            audio_chunk = audio_chunk.squeeze(0)

        # Push samples into sliding window buffer
        window = self.buffer.push(audio_chunk)   # (1, chunk_size)

        # HuBERT features
        features = self.encoder(window)           # (1, T', 1024)

        # FLAME parameter prediction
        dtype = next(self.translator.parameters()).dtype
        features = features.to(dtype=dtype, device=self.device)
        expr_seq, jaw_seq = self.translator(features)  # (1, T', n_expr), (1, T', 3)

        # Take the last frame (most recent)
        expr = expr_seq[0, -1].float()   # (n_expr,)
        jaw = jaw_seq[0, -1].float()     # (3,)

        # Exponential moving average smoothing
        if self._ema_expr is None:
            self._ema_expr = expr
            self._ema_jaw = jaw
        else:
            alpha = self.ema_alpha
            self._ema_expr = alpha * self._ema_expr + (1 - alpha) * expr
            self._ema_jaw = alpha * self._ema_jaw + (1 - alpha) * jaw

        self._last_latency_ms = (time.perf_counter() - t0) * 1000

        return self._ema_expr.clone(), self._ema_jaw.clone()

    def step_from_numpy(self, audio_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convenience wrapper for numpy audio input."""
        chunk = torch.from_numpy(audio_np.astype(np.float32))
        expr, jaw = self.step(chunk)
        return expr.cpu().numpy(), jaw.cpu().numpy()

    def reset(self):
        """Reset buffer and EMA state (call between speakers or sessions)."""
        self.buffer.reset()
        self._ema_expr = None
        self._ema_jaw = None

    @property
    def latency_ms(self) -> float:
        return self._last_latency_ms

    def benchmark(self, n_iters: int = 50, chunk_ms: int = 20) -> dict:
        """
        Measure average latency over n_iters steps.

        Args:
            chunk_ms: Size of each audio chunk in milliseconds (20 ms = 1 HuBERT frame).
        """
        chunk_size = int(16000 * chunk_ms / 1000)
        latencies = []
        for _ in range(n_iters):
            chunk = torch.zeros(chunk_size)
            self.step(chunk)
            latencies.append(self.latency_ms)

        import statistics
        return {
            "mean_ms": statistics.mean(latencies),
            "p95_ms": sorted(latencies)[int(0.95 * len(latencies))],
            "max_ms": max(latencies),
        }
