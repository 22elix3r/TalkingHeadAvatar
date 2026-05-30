from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RuntimeAnimationConfig:
    enabled: bool = True
    fps: float = 30.0
    head_motion_scale: float = 1.0
    eye_motion_scale: float = 1.0
    expression_motion_scale: float = 0.25
    idle_motion_scale: float = 1.0
    blink_rate_per_minute: float = 12.0
    seed: int = 1234


@dataclass
class RuntimeAnimationFrame:
    expression: np.ndarray
    jaw: np.ndarray
    rotation_delta: np.ndarray
    neck_delta: np.ndarray
    eyes_delta: np.ndarray
    translation_delta: np.ndarray


class RuntimeAnimationController:
    """
    Lightweight procedural animation layer for live demos.

    The audio model currently predicts only expression and jaw. This controller
    adds small, bounded FLAME deltas for head, neck, gaze, and micro-expression
    movement so the live avatar does not stay locked to one tracked pose.
    """

    def __init__(
        self,
        n_expr: int,
        config: RuntimeAnimationConfig | None = None,
        expression_basis: np.ndarray | None = None,
    ):
        self.config = config or RuntimeAnimationConfig()
        self.n_expr = int(n_expr)
        self.rng = np.random.default_rng(self.config.seed)
        self.expression_basis = self._coerce_basis(expression_basis)

        self._last_t: float | None = None
        self._speech_level = 0.0
        self._eye_current = np.zeros(2, dtype=np.float32)
        self._eye_target = np.zeros(2, dtype=np.float32)
        self._next_saccade_t = 0.0
        self._next_blink_t = 0.0
        self._blink_start_t: float | None = None

    def _coerce_basis(self, basis: np.ndarray | None) -> np.ndarray:
        if basis is None:
            return np.zeros((0, self.n_expr), dtype=np.float32)
        arr = np.asarray(basis, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out = np.zeros((arr.shape[0], self.n_expr), dtype=np.float32)
        n = min(self.n_expr, arr.shape[1])
        out[:, :n] = arr[:, :n]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-6)

    def step(
        self,
        expression: np.ndarray,
        jaw: np.ndarray,
        has_motion: bool,
        emotion_mode: str = "neutral",
        now: float | None = None,
    ) -> RuntimeAnimationFrame:
        if now is None:
            import time

            now = time.monotonic()
        if self._last_t is None:
            self._last_t = now
            self._next_saccade_t = now + self._rand_range(0.4, 1.2)
            self._next_blink_t = now + self._blink_interval()

        dt = max(1.0 / max(self.config.fps, 1.0), now - self._last_t)
        self._last_t = now

        expr = np.asarray(expression, dtype=np.float32).reshape(-1).copy()
        if expr.shape[0] != self.n_expr:
            fixed = np.zeros(self.n_expr, dtype=np.float32)
            fixed[: min(self.n_expr, expr.shape[0])] = expr[: min(self.n_expr, expr.shape[0])]
            expr = fixed

        jaw_arr = np.zeros(3, dtype=np.float32)
        jaw_in = np.asarray(jaw, dtype=np.float32).reshape(-1)
        jaw_arr[: min(3, jaw_in.shape[0])] = jaw_in[:3]

        speech_raw = min(1.0, float(np.linalg.norm(jaw_arr)) / 0.18) if has_motion else 0.0
        alpha = float(np.exp(-dt / 0.18))
        self._speech_level = alpha * self._speech_level + (1.0 - alpha) * speech_raw

        self._update_gaze(now, dt)
        blink = self._blink_value(now)
        expr = self._apply_expression_motion(expr, now, emotion_mode, blink)

        rotation_delta = self._head_rotation(now)
        neck_delta = rotation_delta * np.array([0.55, 0.45, 0.35], dtype=np.float32)
        eyes_delta = self._eyes_delta(blink)
        translation_delta = self._translation_delta(now)

        return RuntimeAnimationFrame(
            expression=expr,
            jaw=jaw_arr,
            rotation_delta=rotation_delta,
            neck_delta=neck_delta,
            eyes_delta=eyes_delta,
            translation_delta=translation_delta,
        )

    def _head_rotation(self, t: float) -> np.ndarray:
        cfg = self.config
        idle = cfg.idle_motion_scale
        speech = self._speech_level

        pitch = idle * (0.006 * np.sin(0.85 * t) + 0.004 * np.sin(1.7 * t + 1.4))
        yaw = idle * (0.010 * np.sin(0.31 * t + 0.8) + 0.004 * np.sin(0.73 * t))
        roll = idle * (0.004 * np.sin(0.51 * t + 2.0))

        pitch += speech * (0.020 * np.sin(10.5 * t) + 0.010 * np.sin(5.2 * t + 0.5))
        yaw += speech * (0.010 * np.sin(3.7 * t + 1.2))
        roll += speech * (0.004 * np.sin(4.3 * t))

        delta = np.array([pitch, yaw, roll], dtype=np.float32) * cfg.head_motion_scale
        return np.clip(delta, -0.08, 0.08)

    def _translation_delta(self, t: float) -> np.ndarray:
        scale = self.config.idle_motion_scale * self.config.head_motion_scale
        return np.array(
            [
                0.0015 * np.sin(0.47 * t + 1.3),
                0.0012 * np.sin(0.63 * t),
                0.0007 * np.sin(0.41 * t + 2.1),
            ],
            dtype=np.float32,
        ) * scale

    def _update_gaze(self, now: float, dt: float) -> None:
        if now >= self._next_saccade_t:
            self._eye_target = np.array(
                [
                    self._rand_range(-0.028, 0.028),
                    self._rand_range(-0.018, 0.018),
                ],
                dtype=np.float32,
            )
            self._next_saccade_t = now + self._rand_range(0.45, 1.9)

        alpha = 1.0 - float(np.exp(-dt / 0.08))
        self._eye_current += (self._eye_target - self._eye_current) * alpha

    def _eyes_delta(self, blink: float) -> np.ndarray:
        # FLAME stores left/right eye axis-angle triplets. This is gaze motion,
        # not eyelid motion; blink expression is handled separately.
        gaze_x = self._eye_current[0] * self.config.eye_motion_scale
        gaze_y = self._eye_current[1] * self.config.eye_motion_scale
        blink_downcast = 0.018 * blink * self.config.eye_motion_scale
        eye = np.array([gaze_y + blink_downcast, gaze_x, 0.0], dtype=np.float32)
        return np.concatenate([eye, eye]).astype(np.float32)

    def _apply_expression_motion(
        self,
        expr: np.ndarray,
        now: float,
        emotion_mode: str,
        blink: float,
    ) -> np.ndarray:
        if self.expression_basis.shape[0] == 0 or self.config.expression_motion_scale <= 0:
            return expr

        basis = self.expression_basis
        weights = np.zeros(basis.shape[0], dtype=np.float32)
        weights[0] = 0.18 * np.sin(1.25 * now) + 0.10 * self._speech_level
        if basis.shape[0] > 1:
            weights[1] = 0.10 * np.sin(0.73 * now + 1.7)
        if basis.shape[0] > 2:
            weights[2] = 0.08 * np.sin(1.95 * now + 0.4) * (0.3 + self._speech_level)
        if basis.shape[0] > 3:
            weights[3] = 0.12 * blink

        emotion_gain = {
            "neutral": 0.0,
            "engaged": 0.12,
            "emphatic": 0.22,
            "concerned": -0.12,
        }.get(emotion_mode, 0.0)
        weights[0] += emotion_gain

        expr += (weights @ basis) * self.config.expression_motion_scale
        return np.clip(expr, -5.0, 5.0).astype(np.float32)

    def _blink_value(self, now: float) -> float:
        if self.config.blink_rate_per_minute <= 0:
            return 0.0
        if self._blink_start_t is None and now >= self._next_blink_t:
            self._blink_start_t = now

        if self._blink_start_t is None:
            return 0.0

        duration = 0.16
        x = (now - self._blink_start_t) / duration
        if x >= 1.0:
            self._blink_start_t = None
            self._next_blink_t = now + self._blink_interval()
            return 0.0
        return float(np.sin(np.pi * max(0.0, min(1.0, x))) ** 2)

    def _blink_interval(self) -> float:
        mean = 60.0 / max(self.config.blink_rate_per_minute, 1e-6)
        return max(0.8, float(self.rng.exponential(mean)))

    def _rand_range(self, lo: float, hi: float) -> float:
        return float(self.rng.uniform(lo, hi))
