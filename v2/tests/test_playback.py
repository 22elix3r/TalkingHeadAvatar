import asyncio
import struct
from types import SimpleNamespace

import pytest

from talkinghead_v2.audio.playback import PortAudioSink
from talkinghead_v2.contracts import AudioChunk, AudioEOF, ContractError, Envelope, GenerationKey

KEY = GenerationKey("session", "turn", 0)


class Stream:
    def __init__(self, **kwargs):
        self.aborted = self.closed = False
    def start(self): pass
    def abort(self): self.aborted = True
    def close(self): self.closed = True


def chunk(count=8, start=0, sequence=0, key=KEY):
    return AudioChunk(Envelope(key, sequence, "trace"), "segment", 48000, start, tuple((i+1)/10 for i in range(count)))


def callback(sink, frames):
    data = bytearray(frames*4)
    # A past estimated DAC boundary makes drain deterministic in this unit fixture.
    sink._callback(data, frames, SimpleNamespace(currentTime=1.0, outputBufferDacTime=.9), False)
    return struct.unpack(f"<{frames}f", data)


def test_partial_device_callback_preserves_valid_tail_and_does_not_count_padding():
    async def scenario():
        sink = PortAudioSink(stream_factory=Stream, max_buffer_samples=8)
        try:
            await sink.open_device(); sink.start(KEY)
            sink.submit(chunk())
            sink.submit(AudioEOF(Envelope(KEY, 1, "trace"), 48000, 8))
            assert callback(sink, 5) == pytest.approx((.1,.2,.3,.4,.5))
            assert callback(sink, 5) == pytest.approx((.6,.7,.8,0,0))
            assert await sink.drain(KEY)
            report = sink.report()
            assert report["delivered_valid_samples"] == 8 and report["underrun_callbacks"] == 0
            assert report["callback_trace"][-1]["end_sample_exclusive"] == 8
        finally:
            await sink.abort_and_close()
    asyncio.run(scenario())


def test_device_queue_backpressure_preserves_data_and_waits_for_callback():
    async def scenario():
        sink = PortAudioSink(stream_factory=Stream, max_buffer_samples=8)
        try:
            await sink.open_device(); sink.start(KEY); sink.submit(chunk())
            pending = asyncio.create_task(sink.wait_writable(KEY, 4))
            await asyncio.sleep(0)
            assert not pending.done()
            with pytest.raises(ContractError, match="backpressure"):
                sink.submit(chunk(1, start=8, sequence=1))
            assert sink.ledger.next_sample == 8
            callback(sink, 4)
            assert await asyncio.wait_for(pending, 1)
            assert sink.submit(chunk(4, start=8, sequence=1))
            assert sink.peak_queued_samples == 8
        finally:
            await sink.abort_and_close()
    asyncio.run(scenario())


def test_software_cancel_does_not_claim_hardware_flush_and_late_callbacks_cannot_resume():
    async def scenario():
        sink = PortAudioSink(stream_factory=Stream, max_buffer_samples=8)
        try:
            await sink.open_device(); sink.start(KEY)
            stream = sink._stream
            held = chunk()
            sink.submit(held)
            callback(sink, 3)
            assert sink.cancel() == 3 and not stream.aborted
            assert callback(sink, 4) == (0,0,0,0)
            ack = await sink.abort_and_close()
            assert stream.aborted and stream.closed and not ack["physical_flush_qualified"]
            await sink.open_device()
            current = GenerationKey("session", "new", 2)
            sink.start(current)
            assert not sink.submit(held)
            sink.submit(chunk(2, key=current))
            assert callback(sink, 2) == pytest.approx((.1,.2))
        finally:
            await sink.abort_and_close()
    asyncio.run(scenario())


def test_queue_empty_is_not_eof_and_callback_failure_is_reported_instead_of_hanging():
    async def scenario():
        sink = PortAudioSink(stream_factory=Stream)
        try:
            await sink.open_device(); sink.start(KEY)
            sink.submit(chunk(2)); callback(sink, 2); callback(sink, 2)
            assert not sink.finished and sink.underrun_callbacks == 1
            data = bytearray(16)
            sink._callback(data, 4, SimpleNamespace(currentTime=float("nan"), outputBufferDacTime=0), False)
            assert data == bytes(16)
            with pytest.raises(RuntimeError, match="callback failed"):
                await sink.drain(KEY)
        finally:
            await sink.abort_and_close()
    asyncio.run(scenario())
