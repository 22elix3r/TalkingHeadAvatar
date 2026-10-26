"""Bounded callback-driven PortAudio output; software and device stop are distinct."""

import asyncio
from collections import deque
from fractions import Fraction
import struct
import threading
import time

from talkinghead_v2.audio.clock import SampleLedger
from talkinghead_v2.contracts import AudioChunk, AudioEOF, ContractError, GenerationKey, integer


class PortAudioSink:
    cancel_ack_scope = "local_device_queue_only"

    def __init__(self, *, sample_rate=48000, blocksize=960, max_buffer_samples=14400,
                 device=None, stream_factory=None, latency=0.05):
        for name, value in (("sample_rate", sample_rate), ("blocksize", blocksize),
                            ("max_buffer_samples", max_buffer_samples)):
            integer(value, name, 1)
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.max_buffer_samples = max_buffer_samples
        self.device = device
        if type(latency) not in (int, float) or not 0 < latency <= 1:
            raise ContractError("device latency must be a positive number no greater than one second")
        self.latency = latency
        self._factory = stream_factory
        self._stream = None
        self._lock = threading.Lock()
        self._buffers = deque()
        self._queued_samples = 0
        self._loop = None
        self._changed = None
        self._key = None
        self.ledger = None
        self.finished = False
        self._eof = False
        self._delivered = 0
        self._last_dac_end_ns = None
        self._ever_audio = False
        self.trace = deque(maxlen=512)
        self.first_dac_ns = None
        self.underrun_callbacks = 0
        self.portaudio_status_callbacks = 0
        self.portaudio_status_trace = deque(maxlen=64)
        self.peak_queued_samples = 0
        self.device_description = None
        self.closed_ns = None
        self._failure = None

    async def open_device(self):
        if self._stream is not None:
            raise ContractError("output device already open")
        self._loop = asyncio.get_running_loop()
        self._changed = asyncio.Event()
        factory = self._factory
        if factory is None:
            import sounddevice as sd
            factory = sd.RawOutputStream
            self.device_description = dict(sd.query_devices(self.device, "output"))
        try:
            self._stream = await asyncio.to_thread(factory, samplerate=self.sample_rate,
                blocksize=self.blocksize, channels=1, dtype="float32", device=self.device,
                callback=self._callback, latency=self.latency)
            await asyncio.to_thread(self._stream.start)
        except Exception:
            await self.abort_and_close()
            raise

    def start(self, key: GenerationKey, rig=None):
        if self._stream is None:
            raise ContractError("explicitly open the local output device before starting media")
        with self._lock:
            if self._key is not None and not self.finished:
                raise ContractError("finish or cancel old device media before restarting")
            self._key = key
            self.ledger = SampleLedger(key, self.sample_rate)
            self._buffers.clear()
            self._queued_samples = self._delivered = 0
            self._eof = self.finished = self._ever_audio = False
            self._last_dac_end_ns = self.first_dac_ns = None
            self.trace.clear()
            self.underrun_callbacks = self.portaudio_status_callbacks = 0
            self.portaudio_status_trace.clear()
            self.peak_queued_samples = 0
            self._failure = None

    def _notify(self):
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._changed.set)

    def _callback(self, outdata, frames, timing, status):
        try:
            self._fill(outdata, frames, timing, status)
        except Exception as exc:
            outdata[:] = b"\0"*(frames*4)
            with self._lock:
                self._failure = f"audio callback failed: {exc}"
                self._key = None
                self._buffers.clear()
                self._queued_samples = 0
            self._notify()

    def _fill(self, outdata, frames, timing, status):
        before = time.perf_counter_ns()
        # Only map these PortAudio timestamps through the paired currentTime sample.
        current = float(timing.currentTime)
        dac = float(timing.outputBufferDacTime)
        after = time.perf_counter_ns()
        estimated_dac_ns = (before+after)//2 + round((dac-current)*1e9)
        outdata[:] = b"\0"*(frames*4)
        with self._lock:
            if self._key is None:
                return
            if status:
                self.portaudio_status_callbacks += 1
                self.portaudio_status_trace.append({"status": str(status),
                    "output_underflow": bool(getattr(status, "output_underflow", False)),
                    "priming_output": bool(getattr(status, "priming_output", False)),
                    "before_first_audio": not self._ever_audio, "delivered_samples": self._delivered})
            copied = 0
            while self._buffers and copied < frames:
                payload = self._buffers.popleft()
                count = min(len(payload)//4, frames-copied)
                outdata[copied*4:(copied+count)*4] = payload[:count*4]
                copied += count
                if count*4 < len(payload):
                    self._buffers.appendleft(payload[count*4:])
            self._queued_samples -= copied
            if copied:
                start = self._delivered
                self._delivered += copied
                self._ever_audio = True
                self.first_dac_ns = self.first_dac_ns or estimated_dac_ns
                self._last_dac_end_ns = estimated_dac_ns + round(copied*1e9/self.sample_rate)
                self.trace.append({"key_epoch": self._key.epoch, "start_sample": start,
                    "end_sample_exclusive": self._delivered, "estimated_dac_ns": estimated_dac_ns,
                    "portaudio_current_time": current, "portaudio_output_dac_time": dac,
                    "clock_id": "local_monotonic_mapped_from_portaudio",
                    "mapping_sample_interval_ns": after-before})
            if copied < frames and self._ever_audio and not (self._eof and self._queued_samples == 0):
                self.underrun_callbacks += 1
        self._notify()

    async def wait_writable(self, key: GenerationKey, count: int) -> bool:
        if count > self.max_buffer_samples:
            raise ContractError("device chunk exceeds the media-time buffer limit")
        while True:
            with self._lock:
                self._changed.clear()
                if self._failure:
                    raise RuntimeError(self._failure)
                if key != self._key:
                    return False
                if self._queued_samples+count <= self.max_buffer_samples:
                    return True
            await self._changed.wait()

    def submit(self, event) -> bool:
        with self._lock:
            if event.envelope.key != self._key:
                return False
            if isinstance(event, AudioChunk) and self._queued_samples+event.sample_count > self.max_buffer_samples:
                raise ContractError("device backpressure must be awaited, never evicted")
            self.ledger.accept(event)
            if isinstance(event, AudioChunk):
                if event.sample_count:
                    self._buffers.append(struct.pack(f"<{event.sample_count}f", *event.samples))
                    self._queued_samples += event.sample_count
                    self.peak_queued_samples = max(self.peak_queued_samples, self._queued_samples)
            elif isinstance(event, AudioEOF):
                self._eof = True
            return True

    async def drain(self, key: GenerationKey):
        while True:
            with self._lock:
                if self._failure:
                    raise RuntimeError(self._failure)
                if key != self._key:
                    return False
                complete = self._eof and self._queued_samples == 0
                end = self._last_dac_end_ns
                if complete and (end is None or time.perf_counter_ns() >= end):
                    self.finished = True
                    return True
            # This polls completion only. PortAudio's callback/device clock paces PCM.
            await asyncio.sleep(0.005)

    def cancel(self) -> int:
        """Immediate software fence. Does not assert a driver/receiver flush."""
        with self._lock:
            self._key = None
            self._buffers.clear()
            self._queued_samples = 0
            last = self._delivered
        self._notify()
        return last

    async def abort_and_close(self):
        self.cancel()
        stream, self._stream = self._stream, None
        if stream is not None:
            def close():
                try:
                    stream.abort()
                finally:
                    stream.close()
            await asyncio.to_thread(close)
        self.closed_ns = time.perf_counter_ns()
        return {"software_fenced": True, "stream_aborted_and_closed": True,
                "closed_monotonic_ns": self.closed_ns, "physical_flush_qualified": False}

    def report(self):
        with self._lock:
            return self._report()

    def _report(self):
        return {"sample_rate": self.sample_rate, "first_estimated_dac_ns": self.first_dac_ns,
            "last_estimated_dac_end_ns": self._last_dac_end_ns, "delivered_valid_samples": self._delivered,
            "underrun_callbacks": self.underrun_callbacks, "portaudio_status_callbacks": self.portaudio_status_callbacks,
            "portaudio_status_trace": list(self.portaudio_status_trace),
            "requested_device_latency_seconds": self.latency, "callback_blocksize": self.blocksize,
            "peak_queued_samples": self.peak_queued_samples, "max_buffer_samples": self.max_buffer_samples,
            "device": self.device_description, "callback_trace": list(self.trace),
            "callback_failure": self._failure,
            "actual_device_stream": self._factory is None,
            "presentation_boundary": "PortAudio DAC timestamps; acoustic validation separate",
            "timestamp_uncertainty": "pair sampling interval recorded; driver/device timestamp error uncalibrated"}
