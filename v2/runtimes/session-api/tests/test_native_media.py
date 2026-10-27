"""Owned process cancellation/exit tests; no model or GPU initialization."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from talkinghead_v2.contracts import GenerationKey
from talkinghead_v2.server import native_media, transport
from talkinghead_v2.session.device import DeviceLease


async def test_cancellation_during_spawn_reaps_owned_child(tmp_path, monkeypatch):
    script = tmp_path / 'worker'
    script.write_text('#!/usr/bin/python3.12\nimport time\ntime.sleep(30)\n')
    script.chmod(0o700)
    lease = DeviceLease(tmp_path / 'gpu.lock'); lease.acquire('test')
    settings = SimpleNamespace(private_dir=tmp_path, livekit_url='ws://127.0.0.1:17880')
    media = native_media.NativeMedia(settings, 'room', 'session', lease)
    media.config = native_media.NativeMediaConfig(script, tmp_path / 'unused.ptx', 300)
    monkeypatch.setattr(transport, 'grant', lambda *args: 'synthetic-token')
    entered, release = asyncio.Event(), asyncio.Event()
    original = asyncio.create_subprocess_exec
    children = []
    async def delayed(*args, **kwargs):
        entered.set(); await release.wait()
        child = await original(*args, **kwargs); children.append(child)
        return child
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', delayed)
    starting = asyncio.create_task(media.start(GenerationKey('session', 'turn', 1)))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        starting.cancel(); await asyncio.sleep(0)
        assert not starting.done()
        release.set()
        with pytest.raises(asyncio.CancelledError): await asyncio.wait_for(starting, 5)
        assert len(children) == 1 and children[0].returncode is not None
        assert media.state is None and not list(tmp_path.glob('native-epoch-*'))
    finally:
        release.set()
        await media.close(); lease.release()


async def test_external_revocation_during_spawn_cannot_publish_later(tmp_path, monkeypatch):
    script = tmp_path / 'worker'
    script.write_text('#!/usr/bin/python3.12\nimport time\ntime.sleep(30)\n'); script.chmod(0o700)
    lease = DeviceLease(tmp_path / 'gpu.lock'); lease.acquire('test')
    media = native_media.NativeMedia(SimpleNamespace(private_dir=tmp_path, livekit_url='ws://127.0.0.1:17880'), 'room', 'session', lease)
    media.config = native_media.NativeMediaConfig(script, tmp_path / 'unused.ptx', 300)
    monkeypatch.setattr(transport, 'grant', lambda *args: 'synthetic-token')
    entered, release = asyncio.Event(), asyncio.Event()
    original = asyncio.create_subprocess_exec; children = []
    async def delayed(*args, **kwargs):
        entered.set(); await release.wait()
        child = await original(*args, **kwargs); children.append(child); return child
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', delayed)
    starting = asyncio.create_task(media.start(GenerationKey('session', 'turn', 1)))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        media.fence(permanent=True)
        stopping = asyncio.create_task(media.cancel()); await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(stopping, 5)
        with pytest.raises(asyncio.CancelledError): await asyncio.wait_for(starting, 5)
        assert len(children) == 1 and children[0].returncode is not None
        assert media.blocked and media.state is None
        with pytest.raises(PermissionError): await media.start(GenerationKey('session', 'next', 2))
    finally:
        release.set(); await media.close(); lease.release()


async def test_unexpected_worker_exit_fails_eof_instead_of_completing(tmp_path):
    script = tmp_path / 'worker.py'
    script.write_text('print(\'{"kind":"ready","session_id":"session","turn_id":"turn","epoch":4,"audio_sid":"TR_audio","video_sid":"TR_video"}\',flush=True)\n')
    process = await asyncio.create_subprocess_exec('/usr/bin/python3.12', str(script), stdout=asyncio.subprocess.PIPE)
    media = native_media.NativeMedia(SimpleNamespace(private_dir=tmp_path), 'room', 'session', None)
    key = GenerationKey('session', 'turn', 4)
    loop = asyncio.get_running_loop()
    state = {'process': process, 'key': key, 'ready': loop.create_future(), 'eof': loop.create_future()}
    media.state, media.cancelled = state, False
    await media._events(state)
    assert (await state['ready'])['audio_sid'] == 'TR_audio'
    with pytest.raises(RuntimeError, match='exited'): await state['eof']
    assert process.returncode == 0


async def test_exec_failure_removes_private_bootstrap_and_futures(tmp_path, monkeypatch):
    lease = DeviceLease(tmp_path / 'gpu.lock'); lease.acquire('test')
    media = native_media.NativeMedia(SimpleNamespace(private_dir=tmp_path, livekit_url='ws://127.0.0.1:17880'), 'room', 'session', lease)
    media.config = native_media.NativeMediaConfig(tmp_path / 'missing-executable', tmp_path / 'unused.ptx', 300)
    monkeypatch.setattr(transport, 'grant', lambda *args: 'synthetic-token')
    try:
        with pytest.raises(FileNotFoundError): await media.start(GenerationKey('session', 'turn', 1))
        assert media.state is None and not list(tmp_path.glob('native-epoch-*'))
    finally: await media.close(); lease.release()
