import asyncio
from pathlib import Path
import sys
from dataclasses import replace

import pytest

from talkinghead_v2.contracts import AudioChunk, AudioEOF, ContractError, Envelope, GenerationKey, TextSegment
from talkinghead_v2.providers.tts.native import NativeConfig, QwenNativeProvider, TtsError
from talkinghead_v2.session.device import DeviceBusy, DeviceLease


def provider(tmp_path, mode="normal", **kwargs):
    fixture = Path(__file__).parent / "fixtures/fake_qwen_worker.py"
    config = NativeConfig((sys.executable, "-u", str(fixture), mode), "a"*40,
                           "synthetic-protocol-fixture", "tenant", "voice", "synthetic_fixture")
    return QwenNativeProvider(config, diagnostic_log=tmp_path / "private.log", **kwargs)


def segment(key, sequence=0, last=True):
    return TextSegment(Envelope(key, sequence, "trace"), f"segment-{sequence}",
                       "Numbers: 12.50. Café — yes?", "en", "voice", last)


def test_actual_adapter_contract_streams_before_push_completes_and_accounts_multiple_segments(tmp_path):
    async def scenario():
        instance = provider(tmp_path)
        key = GenerationKey("session", "turn", 0)
        try:
            await instance.open_session(key)
            events = []
            first = asyncio.Event()
            async def consume():
                async for event in instance.outputs():
                    events.append(event)
                    first.set()
            consumer = asyncio.create_task(consume())
            producing = asyncio.create_task(instance.push(segment(key, last=False)))
            await asyncio.wait_for(first.wait(), 1)
            assert not producing.done()  # Adapter yields the callback, never waits for the batch result.
            assert await producing
            assert await instance.push(segment(key, sequence=1))
            await asyncio.wait_for(consumer, 1)
            chunks = [event for event in events if isinstance(event, AudioChunk)]
            assert sum(event.sample_count for event in chunks) == 634
            assert chunks[0].start_sample == 0 and chunks[-1].end_sample == 634
            assert all(a.end_sample == b.start_sample for a, b in zip(chunks, chunks[1:]))
            assert [event.segment_id for event in chunks] == ["segment-0"]*3 + ["segment-1"]*3
            assert isinstance(events[-1], AudioEOF) and events[-1].end_sample == 634
            assert instance.stale_packets >= 2
            assert not instance.capabilities.incremental_text_input
            assert instance.capabilities.incremental_audio_output
        finally:
            await instance.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["normal", "backpressure", "hang"])
def test_cancel_fences_packets_and_acknowledges_compute_or_process_exit(tmp_path, mode):
    async def scenario():
        instance = provider(tmp_path, mode)
        old = GenerationKey("session", "old", 0)
        try:
            await instance.open_session(old)
            producing = asyncio.create_task(instance.push(segment(old)))
            if mode == "backpressure":
                await asyncio.wait_for(instance.queue.put_blocked.wait(), 1)
            else:
                first = await asyncio.wait_for(instance.queue.get(old), 1)
                assert isinstance(first, AudioChunk)
            ack = await instance.cancel(timeout=0.02 if mode == "hang" else 1)
            assert ack["compute_stopped"] and not ack["device_flushed"]
            if mode == "hang":
                assert ack["recovery"] == "worker_terminated_cold_reload_required"
            else:
                assert ack["recovery"] == "cooperative"
            assert not await asyncio.wait_for(producing, 1)
            assert instance.queue.item_count == 0
            assert not await instance.push(segment(old))
        finally:
            await instance.close()
    asyncio.run(scenario())


def test_restart_requires_consumed_eof_and_rejects_duplicate_text(tmp_path):
    async def scenario():
        instance = provider(tmp_path)
        key = GenerationKey("session", "turn", 0)
        try:
            await instance.open_session(key)
            assert await instance.push(segment(key))
            with pytest.raises(ContractError, match="consume EOF"):
                await instance.open_session(GenerationKey("session", "next", 1))
            async for _ in instance.outputs():
                pass
            with pytest.raises(ContractError):
                await instance.push(segment(key))
            current = GenerationKey("session", "next", 1)
            await instance.open_session(current)
            assert not await instance.push(segment(key))
            assert await instance.push(segment(current))
            events = [event async for event in instance.outputs()]
            assert events[-1].end_sample == 317
        finally:
            await instance.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["bad-rate", "failure"])
def test_worker_fault_never_becomes_successful_eof_and_releases_worker(tmp_path, mode):
    async def scenario():
        instance = provider(tmp_path, mode)
        key = GenerationKey("session", "turn", 0)
        try:
            await instance.open_session(key)
            with pytest.raises(TtsError):
                await asyncio.wait_for(instance.push(segment(key)), 1)
            assert instance._process is None or instance._process.returncode is not None
            assert instance.queue.item_count == 0
        finally:
            await instance.close()
    asyncio.run(scenario())


def test_device_admission_covers_resident_worker_and_inherited_handle(tmp_path):
    async def scenario():
        template = provider(tmp_path).config
        config = replace(template, device_lease_path=tmp_path / 'gpu-device-0.lock')
        first = QwenNativeProvider(config, diagnostic_log=tmp_path / 'one.log')
        second = QwenNativeProvider(config, diagnostic_log=tmp_path / 'two.log')
        try:
            await first.open_session(GenerationKey('one', 'turn', 0))
            inherited = Path('/proc') / str(first._process.pid) / 'fd' / str(first.device_lease.fileno)
            assert inherited.resolve() == config.device_lease_path
            with pytest.raises(DeviceBusy): await second.open_session(GenerationKey('two', 'turn', 0))
            assert second._process is None
            await first.cancel()
            with pytest.raises(DeviceBusy): await second.open_session(GenerationKey('two', 'turn', 0))
            await first.close()
            await second.open_session(GenerationKey('two', 'turn', 0))
        finally:
            await first.close(); await second.close()
    asyncio.run(scenario())


def test_external_session_lease_is_retained_until_all_consumers_close(tmp_path):
    async def scenario():
        config = replace(provider(tmp_path).config, device_lease_path=tmp_path / 'gpu-device-0.lock')
        lease, competitor = DeviceLease(config.device_lease_path), DeviceLease(config.device_lease_path)
        lease.acquire('actor')
        instance = QwenNativeProvider(config, diagnostic_log=tmp_path / 'one.log', device_lease=lease)
        try:
            await instance.open_session(GenerationKey('actor', 'turn', 0))
            await instance.close()
            with pytest.raises(DeviceBusy): competitor.acquire('other-actor')
            lease.release(); competitor.acquire('other-actor')
        finally:
            await instance.close(); lease.release(); competitor.release()
    asyncio.run(scenario())


def test_cancel_during_process_creation_retires_child_and_denies_duplicate_start(tmp_path, monkeypatch):
    async def scenario():
        config = replace(provider(tmp_path).config, device_lease_path=tmp_path / 'gpu-device-0.lock')
        instance = QwenNativeProvider(config, diagnostic_log=tmp_path / 'one.log')
        launched, deliver = asyncio.Event(), asyncio.Event()
        original = asyncio.create_subprocess_exec
        children = []
        async def delayed(*args, **kwargs):
            child = await original(*args, **kwargs); children.append(child); launched.set()
            await deliver.wait()
            return child
        monkeypatch.setattr(asyncio, 'create_subprocess_exec', delayed)
        opening = asyncio.create_task(instance.open_session(GenerationKey('one', 'turn', 0)))
        try:
            await asyncio.wait_for(launched.wait(), 2)
            with pytest.raises(ContractError, match='startup is already'):
                await instance.open_session(GenerationKey('one', 'other', 1))
            opening.cancel(); deliver.set()
            with pytest.raises(asyncio.CancelledError): await opening
            assert children[0].returncode is not None
            await instance.close()
            other = DeviceLease(config.device_lease_path); other.acquire('other'); other.release()
        finally:
            deliver.set()
            if not opening.done(): opening.cancel()
            await asyncio.gather(opening, return_exceptions=True)
            await instance.close()
    asyncio.run(scenario())
