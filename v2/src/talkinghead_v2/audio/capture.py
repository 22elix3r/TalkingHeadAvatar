"""Explicit local output-monitor capture for presentation evidence; no microphone API."""

import asyncio
import hashlib
import struct
import time


class OutputMonitorCapture:
    def __init__(self, *, max_seconds=120):
        self.max_bytes = max_seconds*48000*4
        self.process = self.task = None
        self.data = bytearray()
        self.monitor = None
        self.armed_ns = None
        self.first_audio_observed_ns = None
        self._partial = b""

    async def start(self):
        probe = await asyncio.create_subprocess_exec("pactl", "get-default-sink", stdout=asyncio.subprocess.PIPE)
        output, _ = await probe.communicate()
        if probe.returncode:
            raise RuntimeError("cannot identify output monitor")
        self.monitor = output.decode().strip()+".monitor"
        self.process = await asyncio.create_subprocess_exec("parec", "--device="+self.monitor, "--format=float32le",
            "--rate=48000", "--channels=1", "--latency-msec=20", stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL)
        self.task = asyncio.create_task(self._collect())

    def arm(self):
        self.armed_ns = time.perf_counter_ns()
        self.first_audio_observed_ns = None
        return self.armed_ns

    async def _collect(self):
        while block := await self.process.stdout.read(32768):
            observed = time.perf_counter_ns()
            self.data.extend(block)
            if len(self.data) > self.max_bytes:
                raise RuntimeError("bounded output-monitor capture exceeded duration limit")
            complete = self._partial+block
            count = len(complete)//4*4
            self._partial = complete[count:]
            if self.armed_ns is not None and self.first_audio_observed_ns is None and any(
                abs(value[0]) > 1e-4 for value in struct.iter_unpack("<f", complete[:count])
            ):
                self.first_audio_observed_ns = observed

    async def close(self, path):
        if self.process is not None:
            if self.process.returncode is None:
                self.process.terminate()
            await self.process.wait()
            await self.task
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.data)
        return {"monitor": self.monitor, "microphone_captured": False, "sample_rate": 48000,
                "samples": len(self.data)//4, "sha256": hashlib.sha256(self.data).hexdigest(),
                "boundary": "output monitor received bytes, local monotonic observation; no acoustic rating"}
