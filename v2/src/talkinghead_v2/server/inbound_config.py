"""Offline verification of an explicitly qualified native audio receiver."""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

from talkinghead_v2.contracts import ContractError

RUST_REVISION = 'bae4df2f3d6c8f0f6ce81a7370c445779f1948d2'
PATCH_SHA256 = 'c740299162fe2764e5d07754f0f79dd94cae64dd76d98753a68f609e26c39330'
CARGO_LOCK_SHA256 = 'f3dbfe24ec94cb2f37eede9962cf50c93673fa44c9c501fcd8c5d820bd93e5a7'
PYTHON_LIVEKIT_VERSION = '1.1.21'


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def private_json(path, limit=65536):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or
        info.st_nlink != 1 or not 0 < info.st_size <= limit):
        raise ContractError('private bounded inbound audio metadata required')
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ContractError('inbound audio metadata object required')
    return data


@dataclass(frozen=True)
class InboundAudioConfig:
    library: Path
    library_sha256: str
    revision: str
    native_frames: int = 25
    python_frames: int = 12

    @classmethod
    def load(cls, path):
        path = Path(path)
        data = private_json(path)
        if (type(data.get('schema_version')) is not int or data['schema_version'] != 1 or data.get('policy') != 'native_audio_overflow_ends_stream' or
            data.get('rust_revision') != RUST_REVISION or data.get('patch_sha256') != PATCH_SHA256 or
            data.get('python_livekit_version') != PYTHON_LIVEKIT_VERSION or
            type(data.get('native_frames')) is not int or data['native_frames'] != 25 or
            type(data.get('python_frames')) is not int or data['python_frames'] != 12):
            raise ContractError('unqualified inbound audio policy')
        records = data.get('files')
        if not isinstance(records, dict) or set(records) != {'library', 'build', 'normal', 'overflow', 'control', 'matrix'}:
            raise ContractError('inbound audio inventory mismatch')
        paths = {}
        for role, record in records.items():
            if not isinstance(record, dict) or set(record) != {'path', 'size', 'sha256'}:
                raise ContractError('inbound audio inventory record mismatch')
            if not isinstance(record['path'], str) or type(record['size']) is not int:
                raise ContractError('inbound audio path/size required')
            if not isinstance(record['sha256'], str) or not re.fullmatch(r'[a-f0-9]{64}', record['sha256']):
                raise ContractError('inbound audio file hash required')
            candidate = path.parent / record['path']
            info = candidate.lstat()
            bound = 128 * 1024 * 1024 if role == 'library' else 65536
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or
                info.st_nlink != 1 or not 0 < info.st_size == record['size'] <= bound or
                file_hash(candidate) != record['sha256']):
                raise ContractError('inbound audio inventory ownership/hash mismatch')
            paths[role] = candidate.resolve(strict=True)
        library_sha = records['library']['sha256']
        build = private_json(paths['build'])
        if (build.get('rust_revision') != RUST_REVISION or build.get('patch_sha256') != PATCH_SHA256 or
            build.get('cargo_lock_sha256') != CARGO_LOCK_SHA256 or build.get('library_sha256') != library_sha or
            type(build.get('native_audio_tests_passed')) is not int or build['native_audio_tests_passed'] != 4):
            raise ContractError('inbound audio build provenance mismatch')
        matrix = private_json(paths['matrix'])
        cases = matrix.get('cases')
        names = {f'{mode}-{index}' for index in range(5) for mode in ('normal', 'overflow')} | {'eviction-control'}
        hashes = matrix.get('ffi_sha256')
        probe = matrix.get('probe_sha256')
        if (type(matrix.get('schema_version')) is not int or matrix['schema_version'] != 1 or matrix.get('status') != 'passed' or
            not isinstance(cases, list) or len(cases) != 11 or
            any(not isinstance(row, dict) or not isinstance(row.get('name'), str) for row in cases) or
            {row['name'] for row in cases} != names or
            any(type(row.get('process_exit')) is not int or row['process_exit'] != 0 or row.get('panic_detected') is not False for row in cases) or
            not isinstance(hashes, dict) or hashes.get('patched') != library_sha or hashes.get('stock') == library_sha or
            not isinstance(probe, str) or not re.fullmatch(r'[a-f0-9]{64}', probe)):
            raise ContractError('clean native audio process-exit matrix missing')
        for row in cases:
            observed = row.get('observation')
            mode = 'eviction_control' if row['name'] == 'eviction-control' else row['name'].rsplit('-', 1)[0]
            if (not isinstance(observed, dict) or observed.get('mode') != mode or
                observed.get('source_sha256') != matrix.get('probe_sha256') or
                observed.get('ffi_sha256') != hashes.get('stock' if mode == 'eviction_control' else 'patched') or
                observed.get('room_connected_at_result') is not True or
                (mode == 'overflow' and (observed.get('native_queue_overflow_eos') is not True or observed.get('stream_ended_before_cleanup') is not True)) or
                (mode != 'overflow' and observed.get('stream_ended_before_cleanup') is not False)):
                raise ContractError('native audio matrix observation mismatch')
        for role in ('normal', 'overflow', 'control'):
            report = private_json(paths[role])
            if not any(row.get('observation') == report for row in cases):
                raise ContractError('inbound audio report lacks a clean parent-observed exit')
            if (type(report.get('schema_version')) is not int or report['schema_version'] != 1 or report.get('real_local_sfu') is not True or
                report.get('synthetic_audio') is not True or type(report.get('native_queue_frames')) is not int or report['native_queue_frames'] != 25 or
                type(report.get('python_queue_frames')) is not int or report['python_queue_frames'] != 12 or report.get('room_connected_at_result') is not True):
                raise ContractError('inbound audio qualification scope mismatch')
            counts = ('received_frames_20ms', 'overflow_warnings_observed', 'eviction_warnings_observed')
            if any(type(report.get(key)) is not int or not 0 <= report[key] <= 100000 for key in counts):
                raise ContractError('inbound audio qualification counters invalid')
            rms = report.get('maximum_rms')
            if type(rms) not in (int, float) or not math.isfinite(rms) or not 0 <= rms <= 32768:
                raise ContractError('inbound audio qualification level invalid')
            if role != 'control' and report.get('ffi_sha256') != library_sha:
                raise ContractError('inbound audio qualification library mismatch')
            if role == 'normal' and (report.get('mode') != 'normal' or report.get('stream_ended_before_cleanup') is not False or
                report.get('received_frames_20ms', 0) < 75 or report.get('maximum_rms', 0) < 100 or
                report.get('overflow_warnings_observed') != 0):
                raise ContractError('healthy inbound audio qualification missing')
            if role == 'overflow' and (report.get('mode') != 'overflow' or report.get('stream_ended_before_cleanup') is not True or
                report.get('native_queue_overflow_eos') is not True or not 0 <= report.get('received_frames_20ms', -1) <= 1 or
                type(report.get('injected_callback_stall_ns')) is not int or report['injected_callback_stall_ns'] < 450_000_000):
                raise ContractError('fail-closed native overflow qualification missing')
            if role == 'control' and (report.get('mode') != 'eviction_control' or report.get('ffi_sha256') == library_sha or
                report.get('stream_ended_before_cleanup') is not False or report.get('eviction_warnings_observed', 0) < 1):
                raise ContractError('unpatched eviction control missing')
        return cls(paths['library'], library_sha, file_hash(path))

    def environment(self, source):
        result = dict(source)
        result['LIVEKIT_LIB_PATH'] = str(self.library)
        result.pop('LD_PRELOAD', None)
        result.pop('LD_AUDIT', None)
        return result
