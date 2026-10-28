"""Sample, cancellation and descriptor contracts without models/SDK/CUDA."""
import asyncio
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from talkinghead_v2.audio.clock import SampleLedger
from talkinghead_v2.audio.resample import RationalResampler
from talkinghead_v2.contracts import AudioChunk, AudioEOF, ContractError, Envelope, GenerationKey
from talkinghead_v2.server.native_media import NativeMedia, NativeMediaConfig, SDK_REVISION


def test_readiness_snapshot_is_numeric_bounded_and_does_not_retain_vendor_text():
    from talkinghead_v2.server.native_media import parse_claire_readiness
    values = [24000, 48000, 24000, 13, 16000, 24000, 0, 3, 900, 1200,
              120, 130, 200, 300, 500, 600, 15, 0, 0, 250]
    line = 'THA_CLAIRE_READINESS_V1 ' + ' '.join(map(str, values))
    snapshot = parse_claire_readiness(['PRIVATE_TOKEN', line, 'Claire GPU owner: private/path'])
    assert snapshot == dict(zip(('valid_samples', 'received_samples', 'queued_audio_samples', 'motion_packets',
        'offered_motion_samples', 'pose_end_samples', 'input_eof', 'owner_stage', 'owner_stage_age_us',
        'owner_progress_age_us', 'push_last_us', 'push_max_us', 'render_last_us', 'render_max_us',
        'capture_last_us', 'capture_max_us', 'source_frames', 'video_skipped', 'owner_failed', 'audio_late_max_us'), values))
    for invalid in (line + ' PRIVATE_TOKEN', line.replace(' 900 ', ' -1 '), line.replace(' 900 ', ' 1.2 '),
                    line.replace(' 900 ', ' 999999999999999999999999 '), line.replace(' 13 ', ' 51 '),
                    line.replace(' 3 ', ' 5 '), line.replace(' 900 ', ' 0900 '), ' ' + line, line + '\nPRIVATE_TOKEN'):
        assert parse_claire_readiness([invalid]) is None
    assert parse_claire_readiness(['THA_CLAIRE_READINESS_V2 0']) is None
    assert 'PRIVATE_TOKEN' not in json.dumps(snapshot)


def descriptor(tmp_path):
    records = {}
    for name in ('executable', 'ptx', 'livekit_library', 'ffi_library'):
        path = tmp_path / name
        path.write_bytes(b'first-party test input')
        path.chmod(0o700 if name == 'executable' else 0o600)
        records[name] = {'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps(dict(schema_version=1, sdk_revision=SDK_REVISION, prebuffer_ms=300, **records)))
    path.chmod(0o600)
    return path


def test_descriptor_requires_pinned_immutable_operator_inputs(tmp_path):
    path = descriptor(tmp_path)
    assert NativeMediaConfig.load(path).prebuffer_ms == 300
    (tmp_path / 'ptx').write_bytes(b'changed kernel')
    with pytest.raises(ContractError, match='hash'): NativeMediaConfig.load(path)


@pytest.mark.parametrize('field,value', [('sdk_revision', 'unknown'), ('prebuffer_ms', True), ('prebuffer_ms', 501), ('prebuffer_ms', 15)])
def test_unqualified_profile_is_not_executed(tmp_path, field, value):
    path = descriptor(tmp_path)
    data = json.loads(path.read_text()); data[field] = value; path.write_text(json.dumps(data))
    with pytest.raises(ContractError): NativeMediaConfig.load(path)


class Pipe:
    def __init__(self): self.packets = []; self.waiting = None
    def write(self, value): self.packets.append(value)
    async def drain(self):
        if self.waiting: await self.waiting.wait()


def media(tmp_path):
    instance = NativeMedia(SimpleNamespace(private_dir=tmp_path), 'room', 'session', None)
    key = GenerationKey('session', 'turn', 3)
    state = {'key': key, 'ledger': SampleLedger(key, 24000), 'resampler': RationalResampler(24000, 48000),
             'sequence': 0, 'sent': 0, 'pending': (), 'eof': asyncio.get_running_loop().create_future(),
             'process': SimpleNamespace(stdin=Pipe(), returncode=0)}
    instance.state, instance.key, instance.cancelled = state, key, False
    return instance, state, key


@pytest.mark.parametrize('sizes', [(1,), (239, 1, 241), (480, 1, 479, 33), (4000, 7001, 999)])
def test_irregular_stream_has_exact_order_and_one_atomic_eof(tmp_path, sizes):
    async def run():
        instance, state, key = media(tmp_path)
        cursor = 0
        for index, count in enumerate(sizes):
            values = tuple(((cursor + offset) % 21 - 10) / 10 for offset in range(count))
            assert await instance.submit(AudioChunk(Envelope(key, index, 'test'), 'segment', 24000, cursor, values))
            cursor += count
        expected_count = cursor * 2
        state['eof'].set_result({'valid_samples': expected_count, 'padding_samples': (-expected_count) % 480 + 960,
            'first_pcm_stream_sample': 0, 'source_clock_late_max_ms': 0., 'queue_peak_samples': expected_count,
            'video_frames': 1, 'video_skipped': 0})
        assert await instance.submit(AudioEOF(Envelope(key, len(sizes), 'test'), 24000, cursor))
        raw, delivered = bytearray(), 0
        for index, packet in enumerate(state['process'].stdin.packets):
            kind, epoch, sequence, start, count = struct.unpack('<cQQQI', packet[:29])
            assert (epoch, sequence, start) == (3, index, delivered)
            assert len(packet[29:]) == count * 2
            assert kind == (b'F' if index == len(state['process'].stdin.packets) - 1 else b'D')
            if kind == b'D': assert count == 960
            raw.extend(packet[29:]); delivered += count
        oracle = RationalResampler(24000, 48000)
        values = tuple((offset % 21 - 10) / 10 for offset in range(cursor))
        expected = oracle.push(values) + oracle.finish()
        assert raw == struct.pack(f'<{expected_count}h', *(int(max(-1., min(1., value)) * 32767) for value in expected))
        assert instance.cursor == expected_count
    asyncio.run(run())


def test_stale_generation_cannot_mutate_active_ledger(tmp_path):
    async def run():
        instance, state, _ = media(tmp_path)
        old = GenerationKey('session', 'old', 2)
        assert not await instance.submit(AudioChunk(Envelope(old, 0, 'test'), 'segment', 24000, 0, (1.,)))
        assert state['ledger'].next_sample == 0 and not state['process'].stdin.packets
    asyncio.run(run())


def test_priority_fence_signals_despite_blocked_audio_pipe(tmp_path):
    async def run():
        instance, state, key = media(tmp_path)
        process = state['process']; process.returncode = None
        signaled = []
        process.terminate = lambda: signaled.append(True)
        pipe = process.stdin; pipe.waiting = asyncio.Event()
        sending = asyncio.create_task(instance._send(state, (0.,) * 960))
        await asyncio.sleep(0)
        assert not sending.done()
        instance.fence(permanent=True)
        assert signaled and instance.blocked
        pipe.waiting.set()
        assert not await sending
        assert instance.cursor == 0
    asyncio.run(run())


def test_worker_eof_cannot_hide_sample_loss(tmp_path):
    async def run():
        instance, state, key = media(tmp_path)
        await instance.submit(AudioChunk(Envelope(key, 0, 'test'), 'segment', 24000, 0, (0.,) * 240))
        state['eof'].set_result({'valid_samples': 479, 'padding_samples': 960})
        with pytest.raises(ContractError, match='accounting'):
            await instance.submit(AudioEOF(Envelope(key, 1, 'test'), 24000, 240))
    asyncio.run(run())


def test_failed_worker_retains_only_bounded_known_diagnostics_and_purges_bootstrap(tmp_path):
    from unittest.mock import AsyncMock
    async def run():
        instance, state, key = media(tmp_path)
        directory = tmp_path / 'native-epoch'
        directory.mkdir(mode=0o700)
        (directory / 'bootstrap.txt').write_text('PRIVATE_SECRET_TOKEN')
        (directory / 'diagnostics.log').write_text('PRIVATE_SECRET_TOKEN\n' * 5000 +
            'THA_CLAIRE_READINESS_V1 24000 48000 24000 13 16000 24000 0 3 900 1200 120 130 200 300 500 600 15 0 0 250\n' +
            'Claire session: Claire motion underrun; retire epoch\n')
        state.update(directory=directory, reader=None, failed=True,
                     process=SimpleNamespace(returncode=1, stdin=None, wait=AsyncMock(return_value=1)))
        state['ready'] = asyncio.get_running_loop().create_future()
        state['ready'].set_result({})
        state['eof'].set_result({})
        await instance._retire()
        retained = list(tmp_path.glob('native-failure-*.json'))
        assert len(retained) == 1 and not directory.exists() and instance.state is None
        assert 'PRIVATE_SECRET_TOKEN' not in retained[0].read_text()
        assert json.loads(retained[0].read_text())['code'] == 'Claire motion underrun; retire epoch'
        assert json.loads(retained[0].read_text())['claire_readiness']['owner_stage'] == 3
        assert retained[0].stat().st_mode & 0o077 == 0
    asyncio.run(run())
