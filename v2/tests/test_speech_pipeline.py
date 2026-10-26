import asyncio
import json
from pathlib import Path
import struct
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from talkinghead_v2.audio.playback import PortAudioSink
from talkinghead_v2.providers.tts import speech
from talkinghead_v2.providers.tts import segments
from talkinghead_v2.providers.tts.native import NativeConfig


@pytest.mark.parametrize("interrupt_after", [None, .005])
def test_complete_adapter_actor_device_path_with_explicit_synthetic_io(tmp_path, monkeypatch, interrupt_after):
    class Capture:
        first_audio_observed_ns = None
        armed_ns = None
        async def start(self): pass
        def arm(self):
            self.armed_ns = time.perf_counter_ns()
            self.first_audio_observed_ns = None
            return self.armed_ns
        async def close(self, path):
            return {"boundary": "synthetic capture fixture", "microphone_captured": False}
    capture = Capture()
    class Stream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]
            self.frames = kwargs["blocksize"]
            self.stopped = threading.Event()
            self.thread = None
        def start(self):
            def drive():
                while not self.stopped.is_set():
                    data = bytearray(self.frames*4)
                    now = time.perf_counter()
                    self.callback(data, self.frames, SimpleNamespace(currentTime=now, outputBufferDacTime=now-.1), False)
                    if capture.armed_ns and capture.first_audio_observed_ns is None and any(
                        abs(value[0]) > 1e-4 for value in struct.iter_unpack("<f", data)
                    ):
                        capture.first_audio_observed_ns = time.perf_counter_ns()
                    time.sleep(.001)
            self.thread = threading.Thread(target=drive)
            self.thread.start()
        def abort(self):
            self.stopped.set()
            if self.thread is not None: self.thread.join()
        def close(self): self.abort()
    fixture = Path(__file__).parent / "fixtures/fake_qwen_worker.py"
    config = NativeConfig((sys.executable, "-u", str(fixture), "normal"), "a"*40,
                          "synthetic-full-flow-fixture", "tenant", "voice", "synthetic_fixture")
    monkeypatch.setattr(speech, "load_native_config", lambda _: config)
    monkeypatch.setattr(speech, "OutputMonitorCapture", lambda: capture)
    monkeypatch.setattr(speech, "PortAudioSink", lambda **kwargs: PortAudioSink(stream_factory=Stream))
    args = SimpleNamespace(runtime=tmp_path / "runtime.json", log=tmp_path / "diagnostic.log", device=None,
                           text="Synthetic integration", interrupt_after=interrupt_after, resume_text="New generation",
                           capture=tmp_path / "capture", report=tmp_path / "report.json")
    report = asyncio.run(speech.run(args))
    assert report["status"] == "passed" and not report["actual_device_playback"]
    completed = report["turns"][-1]
    assert completed["device"]["delivered_valid_samples"] == 634
    assert completed["native"]["samples"] == 317
    assert completed["native_source_emitted_early_pcm"]
    if interrupt_after is not None:
        assert report["turns"][0]["compute_ack"]["compute_stopped"]
        assert report["turns"][0]["task_outcomes"] == ["False", "False", "False"]
        assert completed["epoch"] == 2


def test_device_start_failure_closes_partial_stream():
    instances = []
    class BrokenStream:
        def __init__(self, **kwargs):
            self.closed = False
            instances.append(self)
        def start(self): raise OSError("device unavailable")
        def abort(self): pass
        def close(self): self.closed = True
    async def scenario():
        sink = PortAudioSink(stream_factory=BrokenStream)
        with pytest.raises(OSError, match="unavailable"):
            await sink.open_device()
        assert instances[0].closed and sink._stream is None
    asyncio.run(scenario())


@pytest.mark.parametrize("texts", [[], {}, [{"id": "a", "text": ""}],
    [{"id": "a", "text": "one"}, {"id": "a", "text": "two"}],
    [{"id": "a", "text": "x"*65537}], [{"id": str(i), "text": "one"} for i in range(257)]])
def test_segment_probe_rejects_invalid_input_before_worker_start(tmp_path, monkeypatch, texts):
    path = tmp_path / "texts.json"
    path.write_text(json.dumps(texts))
    def unexpected_load(*args):
        pytest.fail("invalid text must not load a model or start a worker")
    monkeypatch.setattr(segments, "load_native_config", unexpected_load)
    with pytest.raises(ValueError, match="segment probe"):
        asyncio.run(segments.run(SimpleNamespace(texts=path)))
