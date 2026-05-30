"""
Virtual camera frame writer via pyvirtualcam.
"""

from __future__ import annotations

import threading
import time
from queue import Empty, Full, Queue

import numpy as np


class VirtualCameraOutput:
    """
    Writes frames to a v4l2loopback device at a fixed FPS.
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        fps: int = 30,
        device: str = "/dev/video10",
        queue_maxsize: int = 5,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.device = device
        self.frame_queue: Queue[np.ndarray] = Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._startup_event = threading.Event()
        self._startup_error: Exception | None = None
        self._thread: threading.Thread | None = None
        self._last_frame = self._make_idle_frame(0)

    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._startup_event.clear()
        self._startup_error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._startup_event.wait(timeout=5.0):
            self._stop_event.set()
            raise RuntimeError(
                f"Timed out opening virtual camera {self.device}. "
                "Check v4l2loopback setup and device permissions."
            )
        if self._startup_error is not None:
            self._thread = None
            raise self._startup_error

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def submit_frame(self, frame) -> None:
        frame_u8 = self._to_uint8_rgb(frame)
        try:
            self.frame_queue.put_nowait(frame_u8)
        except Full:
            # Drop newest by design (natural decimation policy).
            return

    def _make_idle_frame(self, tick: int = 0) -> np.ndarray:
        """Generate a visible animated idle frame so OBS never sees a pure-black screen."""
        h, w = self.height, self.width
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Animated gradient background
        t = (tick % 120) / 120.0
        x = np.linspace(0, 1, w, dtype=np.float32)
        y = np.linspace(0, 1, h, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        phase = t * 2 * 3.14159
        import math
        r = (0.08 + 0.04 * math.sin(phase)) * xx
        g = (0.05 + 0.04 * math.cos(phase)) * yy
        b = 0.15 + 0.10 * (xx * yy + math.sin(phase) * 0.5)
        frame[:, :, 0] = np.clip(r * 255, 0, 255).astype(np.uint8)
        frame[:, :, 1] = np.clip(g * 255, 0, 255).astype(np.uint8)
        frame[:, :, 2] = np.clip(b * 255, 0, 255).astype(np.uint8)
        # Bright border so OBS users can confirm the feed is live
        bw = max(4, w // 64)
        border_color = (30, 180, 120) if (tick // 30) % 2 == 0 else (180, 80, 30)
        frame[:bw, :] = border_color
        frame[-bw:, :] = border_color
        frame[:, :bw] = border_color
        frame[:, -bw:] = border_color
        return frame

    def _run(self):
        import pyvirtualcam

        idle_tick = 0
        try:
            with pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                device=self.device,
                fmt=pyvirtualcam.PixelFormat.RGB,
            ) as cam:
                print(
                    f"[VirtualCameraOutput] Opened {self.device} "
                    f"{self.width}x{self.height}@{self.fps}fps "
                    f"native_fmt={cam.native_fmt}"
                )
                self._startup_event.set()
                sent_count = 0
                idle_tick = 0
                while not self._stop_event.is_set():
                    try:
                        frame = self.frame_queue.get(timeout=0.05)
                        self._last_frame = frame
                        idle_tick = 0  # reset idle animation when real frame arrives
                    except Empty:
                        # No render frame yet – show animated idle pattern so OBS
                        # never shows a solid-black screen.
                        frame = self._make_idle_frame(idle_tick)
                        idle_tick += 1
                    cam.send(frame)
                    cam.sleep_until_next_frame()
                    sent_count += 1
                    if sent_count % 100 == 0:
                        print(f"[VirtualCameraOutput] Sent {sent_count} frames to {self.device}")
        except Exception as exc:
            err_str = str(exc)
            if "not a video output device" in err_str or "Device or resource busy" in err_str:
                self._startup_error = RuntimeError(
                    f"Failed to open {self.device}: device is busy or in wrong mode. "
                    "Another process is likely already publishing to this node "
                    "(run `fuser -v " + self.device + "` and stop stale producers)."
                )
            else:
                self._startup_error = exc
            self._startup_event.set()

    def _to_uint8_rgb(self, frame) -> np.ndarray:
        if hasattr(frame, "detach") and hasattr(frame, "cpu"):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)

        if frame.ndim != 3:
            raise ValueError(f"Expected 3D frame tensor/array, got shape={frame.shape}")

        # CHW -> HWC
        if frame.shape[0] == 3 and frame.shape[-1] != 3:
            frame = np.transpose(frame, (1, 2, 0))

        if frame.shape[-1] != 3:
            raise ValueError(f"Expected RGB channels=3, got shape={frame.shape}")

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0.0, 1.0)
            frame = (frame * 255.0).astype(np.uint8)

        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(
                f"Frame size mismatch. Expected ({self.height}, {self.width}), got {frame.shape[:2]}"
            )
        return frame
