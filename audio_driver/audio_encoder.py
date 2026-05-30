"""
Audio Encoder — HuBERT Feature Extraction
==========================================
Wraps facebook/hubert-large-ls960-ft for causal-compatible streaming.
HuBERT weights are FROZEN — only the MotionTranslator is trained.

Input:  waveform (B, T) at 16 kHz
Output: features (B, T', 1024) where T' ≈ T / 320

Dependencies: transformers (pip install transformers)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """
    Frozen HuBERT-Large encoder for audio feature extraction.

    The model is loaded lazily on first call to avoid import-time GPU
    allocation (useful when composing multiple pipeline components).
    """

    HUBERT_MODEL = "facebook/hubert-large-ls960-ft"
    FRAME_SHIFT = 320  # samples per HuBERT feature frame at 16 kHz → ~20 ms

    def __init__(
        self,
        model_name: str = HUBERT_MODEL,
        device: str | torch.device = "cuda",
        fp16: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = torch.device(device)
        self.fp16 = fp16
        self._hubert = None  # lazy load

    def _ensure_loaded(self):
        if self._hubert is not None:
            return
        try:
            from transformers import HubertModel
        except ImportError as e:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            ) from e

        print(f"[AudioEncoder] Loading {self.model_name} …")
        self._hubert = HubertModel.from_pretrained(self.model_name)
        self._hubert.eval()

        # Freeze all weights — HuBERT is a universal feature extractor
        for param in self._hubert.parameters():
            param.requires_grad = False

        dtype = torch.float16 if self.fp16 else torch.float32
        self._hubert = self._hubert.to(dtype=dtype, device=self.device)
        print(f"[AudioEncoder] Loaded. Device={self.device}, fp16={self.fp16}")

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) float32/16 tensor at 16 kHz.
                      Values should be in [-1, 1] (raw PCM normalised).
        Returns:
            features: (B, T', 1024) — HuBERT last hidden states.
        """
        self._ensure_loaded()
        waveform = waveform.to(device=self.device, dtype=next(self._hubert.parameters()).dtype)
        outputs = self._hubert(waveform)
        return outputs.last_hidden_state  # (B, T', 1024)

    @property
    def output_dim(self) -> int:
        return 1024

    @property
    def frame_shift_samples(self) -> int:
        return self.FRAME_SHIFT

    def to(self, *args, **kwargs):
        """Override to also move the lazy-loaded HuBERT model."""
        result = super().to(*args, **kwargs)
        if self._hubert is not None:
            self._hubert = self._hubert.to(*args, **kwargs)
        return result


class NullAudioEncoder(nn.Module):
    """
    Drop-in replacement for AudioEncoder that outputs random features.
    Useful for testing the MotionTranslator without downloading HuBERT.
    """

    def __init__(self, output_dim: int = 1024, frame_shift: int = 320):
        super().__init__()
        self._output_dim = output_dim
        self._frame_shift = frame_shift

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        B, T = waveform.shape
        T_prime = T // self._frame_shift
        return torch.randn(B, T_prime, self._output_dim, device=waveform.device)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def frame_shift_samples(self) -> int:
        return self._frame_shift
