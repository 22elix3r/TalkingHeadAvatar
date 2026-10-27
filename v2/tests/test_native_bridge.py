import asyncio
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from talkinghead_v2.contracts import AudioEOF, Envelope, GenerationKey, TextSegment
from talkinghead_v2.providers.tts.native import NativeConfig, QwenNativeProvider


def test_cpp_bridge_isolates_upstream_stdout_from_pcm_protocol(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None or sys.platform != "linux":
        pytest.skip("Linux C++ compiler needed for bridge fault regression; no GPU/models required")
    root = Path(__file__).resolve().parents[1]
    worker = tmp_path / "fixture-worker"
    subprocess.run([compiler, "-std=c++17", "-pthread", "-I", str(root / "tests/fixtures"),
                    str(root / "native/qwen/worker.cpp"), "-o", str(worker)], check=True,
                    capture_output=True, text=True, timeout=30)
    async def scenario():
        config = NativeConfig((str(worker), "fixture-model", "fixture-embedding"), "a"*40,
                              "cpp-protocol-fixture", "tenant", "voice", "synthetic_fixture")
        provider = QwenNativeProvider(config, diagnostic_log=tmp_path / "diagnostics.log")
        try:
            for epoch in range(100):
                key = GenerationKey("fixture-session", str(epoch), epoch)
                await provider.open_session(key)
                assert await provider.push(TextSegment(Envelope(key, 0, "fixture"), "segment",
                                                       "protocol test", "en", "voice", True))
                events = [event async for event in provider.outputs()]
                assert len(events) == 2 and isinstance(events[-1], AudioEOF)
                assert events[-1].end_sample == 2
        finally:
            await provider.close()
    asyncio.run(scenario())
    assert "upstream load stdout diagnostic" in (tmp_path / "diagnostics.log").read_text()
