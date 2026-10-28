"""Pinned local vendor research backend; imported identities are not retargeted."""
import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
from types import MappingProxyType
from typing import Mapping

from talkinghead_v2.assets.artifacts import sha256_file
from talkinghead_v2.contracts import AudioEOF, ContractError
from talkinghead_v2.providers.motion.paired import ClaireInputMux
from talkinghead_v2.server.native_media import NativeMedia, SDK_REVISION

CLAIRE_REVISION = 'a46eafb067adfdd7cc8f5c7941586b309004561c'
CLAIRE_SDK_REVISION = '1ca0f02535ed774f5dbcd724a31cd486368dc783'
CLAIRE_PROTOCOL = 'THA_CLAIRE_MEDIA_2'
RESEARCH_ASSET_ID = 'claire-vendor-research'
REQUIRED = frozenset(('executable', 'ptx', 'bridge_library', 'livekit_library', 'ffi_library', 'sdk_library',
                      'scene', 'scene_descriptor', 'model_descriptor', 'engine', 'engine_descriptor', 'runtime_manifest'))


def owned_descriptor(path):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or
            not 0 < info.st_size <= 128 * 1024):
        raise ContractError('private operator-owned Claire descriptor required')
    return json.loads(path.read_text())


@dataclass(frozen=True)
class ClaireMediaConfig:
    paths: Mapping[str, Path]
    prebuffer_ms: int
    revision: str
    basis_hash: str
    topology_hash: str
    model_hash: str
    gpu_uuid_sha256: str
    driver: str

    @property
    def executable(self): return self.paths['executable']
    @property
    def ptx(self): return self.paths['ptx']

    @classmethod
    def load(cls, path):
        data = owned_descriptor(path)
        if (type(data.get('schema_version')) is not int or data.get('schema_version') != 1 or data.get('profile') != 'claire_vendor_research' or
            data.get('protocol') != CLAIRE_PROTOCOL or data.get('sdk_revision') != SDK_REVISION or data.get('claire_revision') != CLAIRE_REVISION or
            data.get('claire_sdk_revision') != CLAIRE_SDK_REVISION or
            data.get('allowed_use') != 'local_noncommercial_research' or
            data.get('consent_evidence') != 'vendor_assertion' or data.get('human_enrollment') is not False or
            type(data.get('prebuffer_ms')) is not int or not 300 <= data['prebuffer_ms'] <= 500 or data['prebuffer_ms'] % 10):
            raise ContractError('unqualified Claire vendor research profile')
        records = data.get('files', {})
        if not isinstance(records, dict) or not REQUIRED <= records.keys() or not len(records) <= 256:
            raise ContractError('bounded complete Claire runtime inventory required')
        paths, total, hashed = {}, 0, {}
        for name, record in records.items():
            label = PurePosixPath(name)
            if (not name or label.is_absolute() or '..' in label.parts or '\\' in name or
                not isinstance(record, dict) or set(record) != {'path', 'size', 'sha256'} or
                not isinstance(record.get('path'), str) or not 0 < len(record['path']) <= 4096 or
                type(record.get('size')) is not int or
                not 0 <= record['size'] <= 4 * 1024**3):
                raise ContractError('bounded named Claire runtime file required')
            resolved = (path.parent / record['path']).resolve(strict=True)
            info = resolved.stat()
            total += info.st_size
            if (not stat.S_ISREG(info.st_mode) or info.st_uid not in {0, os.getuid()} or info.st_mode & 0o022 or
                info.st_size != record['size'] or total > 8 * 1024**3):
                raise ContractError('Claire runtime file ownership/extent mismatch')
            if name in {'scene_descriptor', 'model_descriptor', 'engine_descriptor', 'runtime_manifest'} and info.st_size > 256 * 1024:
                raise ContractError('bounded Claire metadata required')
            if resolved not in hashed: hashed[resolved] = sha256_file(resolved)
            if hashed[resolved] != record.get('sha256'):
                raise ContractError('Claire runtime file hash mismatch')
            paths[name] = resolved
        if not os.access(paths['executable'], os.X_OK): raise ContractError('Claire executable unavailable')
        engine = json.loads(paths['engine_descriptor'].read_text())
        scene = json.loads(paths['scene_descriptor'].read_text())
        runtime = json.loads(paths['runtime_manifest'].read_text())
        if (engine.get('model_revision') != CLAIRE_REVISION or engine.get('sdk_revision') != CLAIRE_SDK_REVISION or
            engine.get('unknown_external_engine_loaded') is not False or engine.get('build', {}).get('fixed_batch') != 1 or
            engine.get('engine_sha256') != hashed[paths['engine']] or
            engine.get('model_descriptor_sha256') != hashed[paths['model_descriptor']] or
            engine.get('sdk_library_sha256') != hashed[paths['sdk_library']] or
            engine.get('runtime_manifest_sha256') != hashed[paths['runtime_manifest']] or
            scene.get('profile') != 'claire_full_size_research_mesh' or scene.get('vertex_count') != 67122 or
            scene.get('triangle_count') != 134236 or scene.get('geometry_sha256') != hashed[paths['scene']] or
            scene.get('source_geometry_parity_report_sha256') != data.get('basis_hash') or
            scene.get('topology_material_sha256') != data.get('topology_hash')):
            raise ContractError('Claire engine/scene/runtime provenance mismatch')
        for name, expected in engine['model_files'].items():
            if 'model_files/' + name not in paths or hashed[paths['model_files/' + name]] != expected:
                raise ContractError('Claire model inventory binding mismatch')
        for entry in runtime['files']:
            if 'runtime_files/' + entry['path'] not in paths or hashed[paths['runtime_files/' + entry['path']]] != entry['sha256']:
                raise ContractError('Claire isolated runtime binding mismatch')
        model = json.loads(paths['model_descriptor'].read_text())
        bindings = {'networkInfoPath': 'model_files/network_info.json', 'networkPath': 'engine',
                    'emotionDatabasePath': 'model_files/implicit_emo_db.npz', 'modelConfigPath': 'model_files/model_config.json',
                    'modelDataPath': 'model_files/model_data.npz'}
        if set(model) != set(bindings) | {'blendshapePaths'} or set(model['blendshapePaths']) != {'skin', 'tongue'}:
            raise ContractError('qualified Claire model descriptor schema required')
        for field, name in bindings.items():
            if Path(model[field]) != paths[name]: raise ContractError('Claire model dependency binding mismatch')
        for part in ('skin', 'tongue'):
            if set(model['blendshapePaths'][part]) != {'config', 'data'}:
                raise ContractError('qualified Claire blendshape schema required')
            for field, suffix in (('config', '_config.json'), ('data', '.npz')):
                if Path(model['blendshapePaths'][part][field]) != paths['model_files/bs_' + part + suffix]:
                    raise ContractError('Claire blendshape dependency binding mismatch')
        return cls(MappingProxyType(paths), data['prebuffer_ms'], sha256_file(path), data['basis_hash'], data['topology_hash'],
                   engine['engine_sha256'], engine['gpu_uuid_sha256'], engine['driver'])

    def validate_device(self):
        value = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader'], text=True, timeout=5).strip()
        if '\n' in value: raise ContractError('Claire profile qualifies one explicit device only')
        uuid, driver = (part.strip() for part in value.split(','))
        if (hashlib.sha256(uuid.encode()).hexdigest(), driver) != (self.gpu_uuid_sha256, self.driver):
            raise ContractError('Claire engine GPU/driver qualification changed')


class ClaireMedia(NativeMedia):
    runtime_type = ClaireMediaConfig
    bootstrap_version = CLAIRE_PROTOCOL

    def __init__(self, *args, expected_revision=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_revision = expected_revision

    async def connect(self):
        await super().connect()
        if self.expected_revision and self.config.revision != self.expected_revision:
            raise ContractError('Claire runtime changed after asset revision binding')
        cache = self.settings.private_dir / 'claire-cuda-cache'
        cache.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = cache.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ContractError('private project-local Claire CUDA cache required')
        await asyncio.to_thread(self.config.validate_device)

    def _input_state(self, key):
        return {'mux': ClaireInputMux(key)}

    async def _stage_geometry(self, directory):
        # The builtin voice/reference slot is used only with the explicitly
        # pinned vendor scene. Imported identities need their own calibration.
        if self.asset.kind != 'procedural_fixture':
            raise ContractError('Claire vendor research cannot retarget an imported avatar')

    def _command(self, directory):
        return (str(self.config.executable), str(directory / 'bootstrap.txt'), str(self.config.ptx),
                str(self.config.paths['scene']), str(self.config.paths['model_descriptor']))

    def _environment(self):
        environment = dict(os.environ, THA_DEVICE_LEASE_FD=str(self.device_lease.fileno), HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0',
                           CUDA_CACHE_PATH=str(self.settings.private_dir / 'claire-cuda-cache'),
                           LD_LIBRARY_PATH=str(self.config.paths['bridge_library'].parent) + ':' +
                           str(self.config.paths['runtime_manifest'].parent / 'lib') + ':/usr/local/cuda-12.8/lib64')
        environment.pop('LD_PRELOAD', None)
        environment.pop('LD_AUDIT', None)
        environment.pop('THA_CLAIRE_CAPTURE_TRACE', None)
        environment.pop('THA_CLAIRE_PUBLISH_TRACE', None)
        environment.pop('THA_CLAIRE_TRACE_FFI_PATH', None)
        environment.pop('THA_LK_TRUST_EXTERNAL_RATE_CONTROL', None)
        environment.pop('THA_CLAIRE_RATE_PROBE', None)
        environment.pop('THA_CLAIRE_PROBE_BITRATE_BPS', None)
        environment.pop('THA_CLAIRE_RATE_PROBE_SCHEDULE', None)
        return environment

    def _visual_mode(self): return 'claire_vendor_research'

    async def submit(self, event):
        state = self.state
        if self.cancelled or state is None or event.envelope.key != self.key: return False
        packets = await asyncio.to_thread(state['mux'].accept, event)
        for packet in packets:
            if self.cancelled or self.state is not state: return False
            if packet.sequence != state['sequence'] or packet.start_sample != state['sent']:
                raise ContractError('Claire paired sender sequence/sample mismatch')
            state['process'].stdin.write(packet.packed())
            await state['process'].stdin.drain()
            if self.cancelled or self.state is not state: return False
            state['sequence'] += 1
            state['sent'] += len(packet.audio)
            self.cursor = state['sent']
        if not isinstance(event, AudioEOF): return True
        result = await asyncio.wait_for(asyncio.shield(state['eof']), 125)
        if self.cancelled or self.state is not state: return False
        expected = state['mux'].sent_samples
        padding = (-expected) % 480 + 960
        bounded = {'valid_samples': (expected, expected), 'padding_samples': (padding, padding), 'queue_peak_samples': (1, 48000),
                   'motion_queue_peak_packets': (1, 50), 'pose_queue_peak_frames': (1, 32), 'preroll_video_frames': (3, 120000),
                   'motion_eof_padding_frames': (0, 1)}
        if any(type(result.get(name)) is not int or not low <= result[name] <= high for name, (low, high) in bounded.items()):
            raise ContractError('Claire EOF sample/queue accounting mismatch')
        if (result['valid_samples'] != self.cursor or type(result.get('source_video_frames')) is not int or
            type(result.get('video_skipped')) is not int or result['source_video_frames'] < 1 or result['video_skipped'] < 0 or
            result['source_video_frames'] + result['video_skipped'] != (expected + 1599) // 1600):
            raise ContractError('Claire EOF source-frame conservation mismatch')
        self.padding = result['padding_samples']
        state['sender_eof'] = result
        self.sender_metrics = {name: result[name] for name in ('first_pcm_stream_sample', 'source_clock_late_max_ms',
            'queue_peak_samples', 'video_frames', 'video_skipped', 'source_video_frames', 'preroll_video_frames',
            'motion_queue_peak_packets', 'pose_queue_peak_frames', 'motion_eof_padding_frames')}
        return True
