"""
Virtual camera frame writer via pyvirtualcam.
"""

from __future__ import annotations

import threading
import time
from queue import Empty, Full, Queue

import numpy as np
from PIL import Image


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
            if self.width % 2 or self.height % 2:
                raise ValueError(
                    f"I420 virtual camera dimensions must be even, got {self.width}x{self.height}"
                )
            with pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                device=self.device,
                fmt=pyvirtualcam.PixelFormat.I420,
            ) as cam:
                print(
                    f"[VirtualCameraOutput] Opened {self.device} "
                    f"{self.width}x{self.height}@{self.fps}fps "
                    f"input_fmt=I420 native_fmt={cam.native_fmt}"
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
                    cam.send(self._rgb_to_i420(frame))
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
            frame = self._letterbox(frame)
        return frame

    def _letterbox(self, frame: np.ndarray) -> np.ndarray:
        """Resize an RGB frame into the configured camera canvas without distortion."""
        src_h, src_w = frame.shape[:2]
        scale = min(self.width / src_w, self.height / src_h)
        out_w = max(1, int(round(src_w * scale)))
        out_h = max(1, int(round(src_h * scale)))

        image = Image.fromarray(frame, mode="RGB").resize((out_w, out_h), Image.Resampling.BILINEAR)
        canvas = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        x0 = (self.width - out_w) // 2
        y0 = (self.height - out_h) // 2
        canvas[y0 : y0 + out_h, x0 : x0 + out_w] = np.asarray(image, dtype=np.uint8)
        return canvas

    def _rgb_to_i420(self, frame: np.ndarray) -> np.ndarray:
        """Convert HWC RGB uint8 to flat I420/YU12 byte layout."""
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Expected RGB frame shape ({self.height}, {self.width}, 3), got {frame.shape}"
            )
        if self.width % 2 or self.height % 2:
            raise ValueError(
                f"I420 virtual camera dimensions must be even, got {self.width}x{self.height}"
            )

        rgb = frame.astype(np.float32, copy=False)
        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]

        # BT.601 limited-range YUV, matching common V4L2/OBS expectations.
        y = 16.0 + (0.257 * r) + (0.504 * g) + (0.098 * b)
        u = 128.0 - (0.148 * r) - (0.291 * g) + (0.439 * b)
        v = 128.0 + (0.439 * r) - (0.368 * g) - (0.071 * b)

        y_plane = np.clip(y, 16, 235).astype(np.uint8)
        u_plane = self._subsample_420(u)
        v_plane = self._subsample_420(v)
        return np.concatenate(
            [
                y_plane.reshape(-1),
                u_plane.reshape(-1),
                v_plane.reshape(-1),
            ]
        )

    @staticmethod
    def _subsample_420(plane: np.ndarray) -> np.ndarray:
        h, w = plane.shape
        subsampled = (
            plane[0:h:2, 0:w:2]
            + plane[1:h:2, 0:w:2]
            + plane[0:h:2, 1:w:2]
            + plane[1:h:2, 1:w:2]
        ) * 0.25
        return np.clip(subsampled, 16, 240).astype(np.uint8)
