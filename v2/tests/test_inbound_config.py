import json
import os
from pathlib import Path

import pytest

from talkinghead_v2.contracts import ContractError
from talkinghead_v2.server.inbound_config import (
    CARGO_LOCK_SHA256, PATCH_SHA256, PYTHON_LIVEKIT_VERSION, RUST_REVISION,
    InboundAudioConfig, file_hash,
)


def write(path, value):
    path.write_text(json.dumps(value))
    path.chmod(0o600)


@pytest.fixture
def descriptor(tmp_path):
    library = tmp_path / 'ffi.so'
    library.write_bytes(b'not loaded: CPU provenance fixture')
    library.chmod(0o600)
    digest = file_hash(library)
    build = {'rust_revision': RUST_REVISION, 'patch_sha256': PATCH_SHA256,
             'cargo_lock_sha256': CARGO_LOCK_SHA256, 'library_sha256': digest, 'native_audio_tests_passed': 4}
    common = {'schema_version': 1, 'real_local_sfu': True, 'synthetic_audio': True, 'ffi_sha256': digest,
              'source_sha256': 'c' * 64,
              'native_queue_frames': 25, 'python_queue_frames': 12, 'room_connected_at_result': True,
              'received_frames_20ms': 100, 'maximum_rms': 1000, 'overflow_warnings_observed': 0,
              'eviction_warnings_observed': 0, 'stream_ended_before_cleanup': False}
    reports = {'build': build, 'normal': dict(common, mode='normal'),
               'overflow': dict(common, mode='overflow', received_frames_20ms=1, overflow_warnings_observed=1,
                                stream_ended_before_cleanup=True, native_queue_overflow_eos=True, injected_callback_stall_ns=500_000_000),
               'control': dict(common, mode='eviction_control', ffi_sha256='e' * 64, eviction_warnings_observed=1)}
    reports['matrix'] = {'schema_version': 1, 'status': 'passed', 'probe_sha256': 'c' * 64, 'ffi_sha256': {'patched': digest, 'stock': 'e' * 64},
        'cases': [{'name': f'{mode}-{index}', 'process_exit': 0, 'panic_detected': False, 'observation': reports[mode]}
                  for index in range(5) for mode in ('normal', 'overflow')] +
                 [{'name': 'eviction-control', 'process_exit': 0, 'panic_detected': False, 'observation': reports['control']}]}
    for name, data in reports.items():
        write(tmp_path / f'{name}.json', data)
    paths = dict(library=library, **{name: tmp_path / f'{name}.json' for name in reports})
    data = {'schema_version': 1, 'policy': 'native_audio_overflow_ends_stream', 'rust_revision': RUST_REVISION,
            'patch_sha256': PATCH_SHA256, 'python_livekit_version': PYTHON_LIVEKIT_VERSION,
            'native_frames': 25, 'python_frames': 12,
            'files': {role: {'path': path.name, 'size': path.stat().st_size, 'sha256': file_hash(path)} for role, path in paths.items()}}
    path = tmp_path / 'runtime.json'
    write(path, data)
    return path, data


def replace_report(path, data, role, update):
    target = path.parent / data['files'][role]['path']
    report = json.loads(target.read_text())
    report.update(update)
    write(target, report)
    data['files'][role].update(size=target.stat().st_size, sha256=file_hash(target))
    if role in ('normal', 'overflow', 'control'):
        matrix_path = path.parent / data['files']['matrix']['path']
        matrix = json.loads(matrix_path.read_text())
        for row in matrix['cases']:
            if row['observation']['mode'] == report['mode']:
                row['observation'] = report
        write(matrix_path, matrix)
        data['files']['matrix'].update(size=matrix_path.stat().st_size, sha256=file_hash(matrix_path))
    write(path, data)


def test_inbound_config_is_offline_and_child_environment_is_explicit(descriptor):
    path, data = descriptor
    config = InboundAudioConfig.load(path)
    original = {'LD_PRELOAD': 'unqualified.so', 'LD_AUDIT': 'audit.so', 'KEEP': 'value'}
    environment = config.environment(original)
    assert environment == {'LIVEKIT_LIB_PATH': str(path.parent / 'ffi.so'), 'KEEP': 'value'}
    assert original['LD_PRELOAD'] == 'unqualified.so'
    assert config.native_frames == 25 and config.python_frames == 12
    assert config.revision == file_hash(path)
    patch = Path(__file__).resolve().parents[1] / 'patches/livekit-native-audio-fail-closed.patch'
    assert file_hash(patch) == PATCH_SHA256


@pytest.mark.parametrize('role,update', [
    ('normal', {'stream_ended_before_cleanup': True}),
    ('normal', {'maximum_rms': float('nan')}),
    ('overflow', {'stream_ended_before_cleanup': False}),
    ('overflow', {'received_frames_20ms': 5}),
    ('overflow', {'overflow_warnings_observed': True}),
    ('overflow', {'native_queue_overflow_eos': False}),
    ('control', {'eviction_warnings_observed': 0}),
    ('build', {'patch_sha256': '0' * 64}),
    ('matrix', {'status': 'failed'}),
])
def test_changed_qualification_cannot_authorize_a_runtime(descriptor, role, update):
    path, data = descriptor
    replace_report(path, data, role, update)
    with pytest.raises(ContractError):
        InboundAudioConfig.load(path)


def test_changed_or_public_library_is_rejected(descriptor):
    path, data = descriptor
    library = path.parent / 'ffi.so'
    library.chmod(0o644)
    with pytest.raises(ContractError):
        InboundAudioConfig.load(path)
    library.chmod(0o600)
    library.write_bytes(b'changed bytes')
    with pytest.raises(ContractError):
        InboundAudioConfig.load(path)


def test_symlink_inventory_is_rejected_without_reading_target(descriptor):
    path, data = descriptor
    library = path.parent / 'ffi.so'
    library.rename(path.parent / 'retained.so')
    library.symlink_to('retained.so')
    with pytest.raises(ContractError):
        InboundAudioConfig.load(path)
