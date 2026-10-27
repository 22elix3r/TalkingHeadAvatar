"""Operator pinning and async generation fences without SDK/models/GPU."""
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from talkinghead_v2.contracts import AudioChunk, AudioEOF, ContractError, Envelope, GenerationKey
from talkinghead_v2.providers.motion.paired import ClaireInputMux
from talkinghead_v2.server.claire_media import CLAIRE_REVISION, CLAIRE_SDK_REVISION, ClaireMedia, ClaireMediaConfig
from talkinghead_v2.server.native_media import SDK_REVISION


def fixture(tmp_path):
    paths = {}
    def write(role, value):
        path = tmp_path / role
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value, dict) else value)
        path.chmod(0o700 if role == 'executable' else 0o600)
        paths[role] = path
    def digest(role): return hashlib.sha256(paths[role].read_bytes()).hexdigest()
    for name in ('executable', 'ptx', 'bridge_library', 'livekit_library', 'ffi_library', 'sdk_library', 'engine', 'scene'):
        write(name, 'CPU test data, never executed')
    names = ['network_info.json', 'implicit_emo_db.npz', 'model_config.json', 'model_data.npz',
             'bs_skin_config.json', 'bs_skin.npz', 'bs_tongue_config.json', 'bs_tongue.npz']
    for name in names: write('model_files/' + name, 'immutable numeric fixture')
    write('runtime_files/lib/fixture.so', 'not a real library')
    write('runtime_manifest', {'files': [{'path': 'lib/fixture.so', 'sha256': digest('runtime_files/lib/fixture.so')}]})
    write('model_descriptor', {'networkInfoPath': str(paths['model_files/network_info.json']), 'networkPath': str(paths['engine']),
        'emotionDatabasePath': str(paths['model_files/implicit_emo_db.npz']), 'modelConfigPath': str(paths['model_files/model_config.json']),
        'modelDataPath': str(paths['model_files/model_data.npz']), 'blendshapePaths': {part: {
            'config': str(paths['model_files/bs_' + part + '_config.json']), 'data': str(paths['model_files/bs_' + part + '.npz'])} for part in ('skin', 'tongue')}})
    write('engine_descriptor', {'model_revision': CLAIRE_REVISION, 'sdk_revision': CLAIRE_SDK_REVISION,
        'unknown_external_engine_loaded': False, 'build': {'fixed_batch': 1}, 'engine_sha256': digest('engine'),
        'model_descriptor_sha256': digest('model_descriptor'), 'sdk_library_sha256': digest('sdk_library'),
        'runtime_manifest_sha256': digest('runtime_manifest'), 'model_files': {name: digest('model_files/' + name) for name in names},
        'gpu_uuid_sha256': hashlib.sha256(b'GPU-fixture').hexdigest(), 'driver': 'qualified-driver'})
    write('scene_descriptor', {'profile': 'claire_full_size_research_mesh', 'vertex_count': 67122, 'triangle_count': 134236,
        'geometry_sha256': digest('scene'), 'source_geometry_parity_report_sha256': 'a' * 64, 'topology_material_sha256': 'b' * 64})
    def records(): return {role: {'path': str(path), 'size': path.stat().st_size, 'sha256': digest(role)} for role, path in paths.items()}
    data = {'schema_version': 1, 'profile': 'claire_vendor_research', 'sdk_revision': SDK_REVISION,
        'claire_revision': CLAIRE_REVISION, 'claire_sdk_revision': CLAIRE_SDK_REVISION, 'prebuffer_ms': 300,
        'allowed_use': 'local_noncommercial_research', 'consent_evidence': 'vendor_assertion', 'human_enrollment': False,
        'basis_hash': 'a' * 64, 'topology_hash': 'b' * 64, 'files': records()}
    descriptor = tmp_path / 'runtime.json'
    def save(): data['files'] = records(); descriptor.write_text(json.dumps(data)); descriptor.chmod(0o600)
    save()
    return descriptor, data, paths, save


def test_pinning_is_offline_and_separately_checks_device_identity(tmp_path, monkeypatch):
    descriptor, _, _, _ = fixture(tmp_path)
    query = Mock(return_value='GPU-fixture, qualified-driver\n')
    monkeypatch.setattr('subprocess.check_output', query)
    config = ClaireMediaConfig.load(descriptor)
    query.assert_not_called()
    config.validate_device()
    query.return_value = 'GPU-fixture, changed-driver'
    with pytest.raises(ContractError, match='GPU/driver'): config.validate_device()
    query.return_value = 'GPU-fixture, qualified-driver\nGPU-other, qualified-driver'
    with pytest.raises(ContractError, match='one explicit device'): config.validate_device()
    with pytest.raises(TypeError): config.paths['engine'] = Path('/unqualified')


@pytest.mark.parametrize('field,value', [('schema_version', True), ('prebuffer_ms', True), ('prebuffer_ms', 299), ('prebuffer_ms', 305),
    ('claire_revision', 'unknown'), ('claire_sdk_revision', 'unknown'), ('allowed_use', 'unrestricted'),
    ('consent_evidence', 'human_verified'), ('human_enrollment', True)])
def test_scope_and_runtime_cannot_silently_change(tmp_path, field, value):
    descriptor, data, _, save = fixture(tmp_path)
    data[field] = value
    save()
    with pytest.raises(ContractError, match='profile'): ClaireMediaConfig.load(descriptor)


def test_changed_numeric_model_data_and_unsafe_permissions_fail_before_execution(tmp_path):
    descriptor, _, paths, _ = fixture(tmp_path)
    paths['model_files/model_data.npz'].write_text('tampered numeric payload')
    with pytest.raises(ContractError, match='extent|hash'): ClaireMediaConfig.load(descriptor)
    descriptor.chmod(0o644)
    with pytest.raises(ContractError, match='private operator'): ClaireMediaConfig.load(descriptor)


def test_rehashed_descriptor_cannot_redirect_sdk_to_an_unlisted_model(tmp_path):
    descriptor, _, paths, save = fixture(tmp_path)
    model = json.loads(paths['model_descriptor'].read_text())
    model['modelDataPath'] = str(tmp_path / 'unlisted.npz')
    paths['model_descriptor'].write_text(json.dumps(model))
    engine = json.loads(paths['engine_descriptor'].read_text())
    engine['model_descriptor_sha256'] = hashlib.sha256(paths['model_descriptor'].read_bytes()).hexdigest()
    paths['engine_descriptor'].write_text(json.dumps(engine))
    save()
    with pytest.raises(ContractError, match='dependency binding'): ClaireMediaConfig.load(descriptor)


class Pipe:
    def __init__(self): self.packets = []; self.entered = asyncio.Event(); self.release = None
    def write(self, packet): self.packets.append(packet)
    async def drain(self):
        self.entered.set()
        if self.release: await self.release.wait()


def media(tmp_path):
    instance = ClaireMedia(SimpleNamespace(private_dir=tmp_path), 'room', 'session', None)
    key = GenerationKey('session', 'turn', 4)
    state = {'key': key, 'mux': ClaireInputMux(key), 'sequence': 0, 'sent': 0,
             'eof': asyncio.get_running_loop().create_future(), 'process': SimpleNamespace(stdin=Pipe(), returncode=0)}
    instance.state, instance.key, instance.cancelled = state, key, False
    return instance, state, key


@pytest.mark.parametrize('corrupt', [None, 'padding_samples', 'valid_samples', 'source_video_frames', 'pose_queue_peak_frames'])
def test_sender_eof_cannot_hide_loss_or_queue_overflow(tmp_path, corrupt):
    async def run():
        instance, state, key = media(tmp_path)
        await instance.submit(AudioChunk(Envelope(key, 0, 'test'), 'segment', 24000, 0, (.1,) * 721))
        result = dict(valid_samples=1442, padding_samples=1438, source_video_frames=1, video_skipped=0, queue_peak_samples=1442,
            motion_queue_peak_packets=2, pose_queue_peak_frames=1, preroll_video_frames=3, first_pcm_stream_sample=4800,
            source_clock_late_max_ms=0., video_frames=4)
        if corrupt: result[corrupt] += 1 if corrupt != 'pose_queue_peak_frames' else 32
        state['eof'].set_result(result)
        eof = AudioEOF(Envelope(key, 1, 'test'), 24000, 721)
        if corrupt:
            with pytest.raises(ContractError, match='accounting|conservation'): await instance.submit(eof)
        else:
            assert await instance.submit(eof)
            assert instance.cursor == 1442 and instance.padding == 1438
            assert state['process'].stdin.packets[-1][:1] == b'F'
    asyncio.run(run())


def test_priority_fence_prevents_late_cursor_commit_through_blocked_pipe(tmp_path):
    async def run():
        instance, state, key = media(tmp_path)
        pipe = state['process'].stdin
        pipe.release = asyncio.Event()
        state['process'].returncode = None
        state['process'].terminate = Mock()
        sending = asyncio.create_task(instance.submit(AudioChunk(Envelope(key, 0, 'test'), 'segment', 24000, 0, (.2,) * 960)))
        await asyncio.wait_for(pipe.entered.wait(), 2)
        instance.fence(permanent=True)
        state['process'].terminate.assert_called_once()
        pipe.release.set()
        assert not await sending and instance.cursor == 0
        assert instance.blocked
    asyncio.run(run())


def test_imported_identity_is_rejected_and_child_environment_does_not_mutate_parent(tmp_path, monkeypatch):
    async def run():
        instance, _, _ = media(tmp_path)
        instance.asset = SimpleNamespace(kind='human_consented')
        with pytest.raises(ContractError, match='cannot retarget'): await instance._stage_geometry(tmp_path)
        instance.device_lease = SimpleNamespace(fileno=7)
        instance.config = SimpleNamespace(paths={'runtime_manifest': tmp_path / 'runtime/runtime.json'})
        monkeypatch.setenv('LD_PRELOAD', 'unqualified.so')
        environment = instance._environment()
        assert 'LD_PRELOAD' not in environment and environment['THA_DEVICE_LEASE_FD'] == '7'
        assert environment['HF_HUB_OFFLINE'] == '1'
        import os
        assert os.environ['LD_PRELOAD'] == 'unqualified.so'
    asyncio.run(run())


def test_cross_session_start_and_changed_serving_revision_fail_before_publication(tmp_path, monkeypatch):
    async def run():
        instance, state, _ = media(tmp_path)
        with pytest.raises(ContractError, match='session identity'):
            await instance.start(GenerationKey('other-session', 'turn', 4))
        assert instance.state is state and not instance.cancelled
        instance.state = None
        descriptor, _, _, _ = fixture(tmp_path)
        config = ClaireMediaConfig.load(descriptor)
        instance.settings = SimpleNamespace(native_media_runtime=descriptor)
        instance.device_lease = SimpleNamespace(fileno=7)
        instance.runtime_type = SimpleNamespace(load=lambda path: config)
        instance.expected_revision = 'different-approved-revision'
        hardware = Mock()
        monkeypatch.setattr(ClaireMediaConfig, 'validate_device', hardware)
        with pytest.raises(ContractError, match='asset revision'): await instance.connect()
        hardware.assert_not_called()
    asyncio.run(run())
