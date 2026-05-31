from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FlameMotionFrame:
    """One normalized FLAME motion prediction from the audio driver."""

    expression: np.ndarray
    jaw: np.ndarray
    rotation: np.ndarray
    neck: np.ndarray
    eyes: np.ndarray
    translation: np.ndarray

    @classmethod
    def zeros(cls, n_expr: int = 100) -> "FlameMotionFrame":
        return cls(
            expression=np.zeros(n_expr, dtype=np.float32),
            jaw=np.zeros(3, dtype=np.float32),
            rotation=np.zeros(3, dtype=np.float32),
            neck=np.zeros(3, dtype=np.float32),
            eyes=np.zeros(6, dtype=np.float32),
            translation=np.zeros(3, dtype=np.float32),
        )

    @classmethod
    def from_channels(
        cls,
        channels: dict[str, np.ndarray],
        n_expr: int = 100,
    ) -> "FlameMotionFrame":
        frame = cls.zeros(n_expr=n_expr)
        if "expr" in channels:
            frame.expression = _coerce(channels["expr"], n_expr)
        if "jaw" in channels:
            frame.jaw = _coerce(channels["jaw"], 3)
        if "rotation" in channels:
            frame.rotation = _coerce(channels["rotation"], 3)
        if "neck" in channels:
            frame.neck = _coerce(channels["neck"], 3)
        if "eyes" in channels:
            frame.eyes = _coerce(channels["eyes"], 6)
        if "translation" in channels:
            frame.translation = _coerce(channels["translation"], 3)
        return frame

    def scaled(
        self,
        expr_scale: float = 1.0,
        jaw_scale: float = 1.0,
        rotation_scale: float = 1.0,
        neck_scale: float = 1.0,
        eyes_scale: float = 1.0,
        translation_scale: float = 1.0,
    ) -> "FlameMotionFrame":
        return FlameMotionFrame(
            expression=np.clip(self.expression * expr_scale, -5.0, 5.0).astype(np.float32),
            jaw=np.clip(self.jaw * jaw_scale, -0.45, 0.45).astype(np.float32),
            rotation=np.clip(self.rotation * rotation_scale, -0.18, 0.18).astype(np.float32),
            neck=np.clip(self.neck * neck_scale, -0.18, 0.18).astype(np.float32),
            eyes=np.clip(self.eyes * eyes_scale, -0.25, 0.25).astype(np.float32),
            translation=np.clip(self.translation * translation_scale, -0.08, 0.08).astype(np.float32),
        )


def _coerce(value: np.ndarray, width: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    out = np.zeros(width, dtype=np.float32)
    out[: min(width, arr.shape[0])] = arr[:width]
    return out
