"""Opt-in pinned C++/NVENC static reference, with one owned child per epoch.

Sender EOF is not a receiver presentation acknowledgment. Fresh publications
remain the conservative browser resume boundary after priority cancellation.
"""
import asyncio
from dataclasses import dataclass
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import tempfile

from talkinghead_v2.assets.avatar import load_avatar
from talkinghead_v2.audio.clock import SampleLedger
from talkinghead_v2.audio.resample import RationalResampler
from talkinghead_v2.contracts import AudioChunk, ContractError
from talkinghead_v2.rendering.projected import project_neutral

SDK_REVISION = 'df8319db2f459d194ae0bb7a5d04920ce5641f81'
IDENTIFIER = re.compile(r'[A-Za-z0-9_-]{1,100}\Z')
CLAIRE_READINESS_FIELDS = (
    ('valid_samples', 5760000), ('received_samples', 5760000), ('queued_audio_samples', 48000),
    ('motion_packets', 50), ('offered_motion_samples', 1920000), ('pose_end_samples', 5763200),
    ('input_eof', 1), ('owner_stage', 4), ('owner_stage_age_us', 3600000000),
    ('owner_progress_age_us', 3600000000), ('push_last_us', 3600000000), ('push_max_us', 3600000000),
    ('render_last_us', 3600000000), ('render_max_us', 3600000000), ('capture_last_us', 3600000000),
    ('capture_max_us', 3600000000), ('source_frames', 108000), ('video_skipped', 108000),
    ('owner_failed', 1), ('audio_late_max_us', 3600000000))


def parse_claire_readiness(lines):
    """Fault-only numeric schema; never copy arbitrary native/vendor diagnostics."""
    for line in reversed(lines):
        if not line.startswith('THA_CLAIRE_READINESS_V1 ') or len(line) > 512:
            continue
        values = line.removeprefix('THA_CLAIRE_READINESS_V1 ').split(' ')
        if len(values) != len(CLAIRE_READINESS_FIELDS) or any(
                re.fullmatch(r'0|[1-9][0-9]{0,9}', value) is None for value in values):
            continue
        numeric = list(map(int, values))
        if all(value <= bound for value, (_, bound) in zip(numeric, CLAIRE_READINESS_FIELDS)):
            return {name: value for (name, _), value in zip(CLAIRE_READINESS_FIELDS, numeric)}
    return None


def private_write(path, value):
    with open(path, 'xb', opener=lambda name, flags: os.open(name, flags | os.O_NOFOLLOW, 0o600)) as target:
        target.write(value)


@dataclass(frozen=True)
class NativeMediaConfig:
    executable: Path
    ptx: Path
    prebuffer_ms: int

    @classmethod
    def load(cls, path):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise ContractError('operator-owned native media descriptor required')
        data = json.loads(path.read_text())
        if (data.get('schema_version') != 1 or data.get('sdk_revision') != SDK_REVISION or
            type(data.get('prebuffer_ms')) is not int or not 10 <= data['prebuffer_ms'] <= 500 or data['prebuffer_ms'] % 10):
            raise ContractError('unqualified native media runtime')
        paths = {}
        for name in ('executable', 'ptx', 'livekit_library', 'ffi_library'):
            record = data[name]
            resolved = (path.parent / record['path']).resolve(strict=True)
            file_info = resolved.stat()
            if (not stat.S_ISREG(file_info.st_mode) or file_info.st_uid != os.getuid() or file_info.st_mode & 0o022 or
                hashlib.sha256(resolved.read_bytes()).hexdigest() != record['sha256']):
                raise ContractError('native runtime ownership/hash mismatch')
            paths[name] = resolved
        if not os.access(paths['executable'], os.X_OK): raise ContractError('native media executable unavailable')
        return cls(paths['executable'], paths['ptx'], data['prebuffer_ms'])


class NativeMedia:
    runtime_type = NativeMediaConfig
    bootstrap_version = 'THA_NATIVE_MEDIA_1'

    def __init__(self, settings, room_name, sid, device_lease):
        self.settings, self.room_name, self.sid = settings, room_name, sid
        self.device_lease = device_lease
        self.key = None
        self.cancelled = True
        self.blocked = False
        self.cursor = self.padding = 0
        self.state = None
        self.config = None
        self.cleanup_lock = asyncio.Lock()
        self.sender_metrics = None
        self.asset = load_avatar(Path(str(files('talkinghead_v2') / 'fixtures/p03-avatar/avatar.json')), tenant_id='procedural')
        self.renderer = None  # Assigned by enrollment flow; native projection uses the same immutable asset.

    async def connect(self):
        if self.device_lease is None: raise ContractError('native media requires managed device admission')
        _ = self.device_lease.fileno
        self.config = await asyncio.to_thread(self.runtime_type.load, self.settings.native_media_runtime)

    def _input_state(self, key):
        return {'ledger': SampleLedger(key, 24000), 'resampler': RationalResampler(24000, 48000)}

    async def _stage_geometry(self, directory):
        geometry = await asyncio.to_thread(project_neutral, self.asset, 512)
        if self.cancelled: raise asyncio.CancelledError
        private_write(directory / 'mesh.bin', geometry.packed())

    def _command(self, directory):
        return (str(self.config.executable), str(directory / 'bootstrap.txt'), str(self.config.ptx), str(directory / 'mesh.bin'))

    def _environment(self):
        return None

    def _visual_mode(self):
        return 'static_procedural_reference' if self.asset.kind == 'procedural_fixture' else 'static_mesh_reference'

    async def start(self, key):
        if key.session_id != self.sid: raise ContractError('native media session identity mismatch')
        if not all(IDENTIFIER.fullmatch(value) for value in (key.session_id, key.turn_id)):
            raise ContractError('bounded native session/turn identifiers required')
        await self.cancel()
        if self.blocked: raise PermissionError('media_authorization_revoked')
        from talkinghead_v2.server.transport import grant
        self.key, self.cancelled = key, False
        self.cursor = self.padding = 0
        self.sender_metrics = None
        directory = Path(tempfile.mkdtemp(prefix='native-epoch-', dir=self.settings.private_dir))
        loop = asyncio.get_running_loop()
        state = {'directory': directory, 'process': None, 'reader': None, 'ready': loop.create_future(),
                 'eof': loop.create_future(), 'key': key, 'sequence': 0, 'sent': 0, 'pending': (),
                 **self._input_state(key)}
        # Observe failures even when cancellation prevents a waiter from reaching a future.
        for name in ('ready', 'eof'):
            state[name].add_done_callback(lambda future: None if future.cancelled() else future.exception())
        self.state = state
        try:
            await self._stage_geometry(directory)
            if self.cancelled: raise asyncio.CancelledError
            token = grant(self.settings, self.room_name, 'media-' + self.sid, 'media')
            private_write(directory / 'bootstrap.txt', ('\n'.join((self.bootstrap_version, self.settings.livekit_url,
                token, key.session_id, key.turn_id, str(key.epoch), str(self.config.prebuffer_ms))) + '\n').encode())
            log = open(directory / 'diagnostics.log', 'xb', opener=lambda name, flags: os.open(name, flags, 0o600))
            try:
                opening = state['opening'] = asyncio.create_task(asyncio.create_subprocess_exec(*self._command(directory),
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log,
                    pass_fds=(self.device_lease.fileno,), env=self._environment(), limit=16384))
                try: state['process'] = await asyncio.shield(opening)
                except asyncio.CancelledError:
                    state['process'] = await opening
                    raise
            finally: log.close()
            process = state['process']
            if self.cancelled or self.state is not state: raise asyncio.CancelledError
            process.stdin.transport.set_write_buffer_limits(high=8192, low=4096)
            state['reader'] = asyncio.create_task(self._events(state))
            binding = await asyncio.wait_for(asyncio.shield(state['ready']), 20)
            if self.cancelled or self.state is not state: raise asyncio.CancelledError
            return {**{name: binding[name] for name in ('audio_sid', 'video_sid', 'publisher', 'epoch')},
                    'visual_mode': self._visual_mode(),
                    'clock_mapping': 'new_track_per_epoch', 'source_sample_rate': 48000,
                    'transport': 'livekit_cpp_preencoded', 'source_clock': 'cpp_monotonic',
                    'prebuffer_ms': self.config.prebuffer_ms}
        except BaseException:
            await self.cancel()
            raise

    async def _events(self, state):
        process, key = state['process'], state['key']
        try:
            while line := await process.stdout.readline():
                message = json.loads(line)
                if message.get('kind') == 'error': raise RuntimeError('native_media_failed; inspect private diagnostics')
                if (message.get('session_id'), message.get('turn_id'), message.get('epoch')) != (key.session_id, key.turn_id, key.epoch):
                    raise ContractError('native worker event generation mismatch')
                kind = message.get('kind')
                if kind == 'ready':
                    if state['ready'].done() or not all(IDENTIFIER.fullmatch(message[name]) for name in ('audio_sid', 'video_sid')):
                        raise ContractError('duplicate/invalid native publication binding')
                    state['ready'].set_result(message)
                elif kind == 'eof':
                    if not state['ready'].done() or state['eof'].done() or message.get('presentation_ack') is not False:
                        raise ContractError('invalid native EOF ordering')
                    state['eof'].set_result(message)
                else: raise ContractError('unknown native worker event')
            await process.wait()
            if not self.cancelled and self.state is state: raise RuntimeError('native worker exited before cancellation')
        except Exception as error:
            state['failed'] = True
            for name in ('ready', 'eof'):
                if not state[name].done(): state[name].set_exception(error)
            if process.returncode is None:
                try: process.terminate()
                except ProcessLookupError: pass

    async def _send(self, state, samples, *, final=False):
        if self.cancelled or self.state is not state: return False
        # PCM conversion clips only amplitude; it never changes sample count/order.
        pcm = struct.pack(f'<{len(samples)}h', *(int(max(-1., min(1., value)) * 32767) for value in samples))
        header = struct.pack('<cQQQI', b'F' if final else b'D', state['key'].epoch,
                             state['sequence'], state['sent'], len(samples))
        process = state['process']
        process.stdin.write(header + pcm)
        await process.stdin.drain()
        if self.cancelled or self.state is not state: return False
        state['sequence'] += 1
        state['sent'] += len(samples)
        self.cursor = state['sent']
        return True

    async def submit(self, event):
        state = self.state
        if self.cancelled or state is None or event.envelope.key != self.key: return False
        state['ledger'].accept(event)
        if isinstance(event, AudioChunk):
            if event.end_sample > 24000 * 120: raise ContractError('native turn exceeds 120-second input bound')
            values = await asyncio.to_thread(state['resampler'].push, event.samples)
            final = False
        else:
            if event.valid_end_sample != event.end_sample:
                raise ContractError('native Qwen transport requires whole source-sample EOF')
            values = await asyncio.to_thread(state['resampler'].finish)
            final = True
        if self.cancelled or self.state is not state: return False
        pending = state['pending'] + values
        # Retain at least the last partial quantum until EOF, atomically flagging
        # its end instead of racing an empty queue with a subsequent EOF packet.
        while len(pending) >= 960:
            if not await self._send(state, pending[:960]): return False
            pending = pending[960:]
        state['pending'] = pending
        if not final: return True
        if not await self._send(state, pending, final=True): return False
        state['pending'] = ()
        result = await asyncio.wait_for(asyncio.shield(state['eof']), 125)
        if self.cancelled or self.state is not state: return False
        if (type(result.get('valid_samples')) is not int or result['valid_samples'] != state['resampler'].output_count or
            result['valid_samples'] != self.cursor or type(result.get('padding_samples')) is not int or
            not 960 <= result['padding_samples'] <= 1439):
            raise ContractError('native EOF sample accounting mismatch')
        self.padding = result['padding_samples']
        state['sender_eof'] = result
        self.sender_metrics = {name: result[name] for name in ('first_pcm_stream_sample', 'source_clock_late_max_ms',
            'queue_peak_samples', 'video_frames', 'video_skipped')}
        return True

    def fence(self, *, permanent=False):
        self.blocked = self.blocked or permanent
        self.cancelled = True
        state = self.state
        if state and state['process'] and state['process'].returncode is None:
            # A separate signal bypasses a producer blocked on a full PCM pipe.
            try: state['process'].terminate()
            except ProcessLookupError: pass

    async def cancel(self):
        self.fence()
        async with self.cleanup_lock:
            await self._retire()

    async def _retire(self):
        state = self.state
        if state is None: return
        if state.get('opening'):
            # Also cover an external revocation during subprocess creation.
            try: state['process'] = await asyncio.shield(state['opening'])
            except Exception: pass  # Failed exec created no child; still purge its private bootstrap.
            self.fence()
        process = state['process']
        if process:
            try: await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                try: process.kill()
                except ProcessLookupError: pass
                await process.wait()
            if state['reader']: await state['reader']
            if process.stdin:
                process.stdin.close()
                try: await process.stdin.wait_closed()
                except (BrokenPipeError, ConnectionResetError): pass
            if state.get('failed') or (process.returncode is not None and process.returncode > 0):
                self._retain_failure(state)
        for name in ('ready', 'eof'):
            if not state[name].done(): state[name].cancel()
        shutil.rmtree(state['directory'])
        if self.state is state: self.state = None

    def _retain_failure(self, state):
        # Retain fixed fault strings and a bounded numeric readiness schema,
        # never vendor stderr, credentials, model paths, PCM or exception text.
        allowed = {'active Claire audio underrun; retire epoch', 'Claire motion underrun; retire epoch',
                   'claire_input_protocol_failure', 'claire_gpu_media_failure', 'Claire encoded frame rejected',
                   'local SFU connection failed', 'native_gpu_media_failure', 'pcm_protocol_failure',
                   'active audio underrun; no silent speech deletion'}
        try:
            with open(state['directory'] / 'diagnostics.log', 'rb', opener=lambda name, flags: os.open(name, flags | os.O_NOFOLLOW)) as source:
                source.seek(0, os.SEEK_END)
                source.seek(max(0, source.tell() - 16384))
                lines = source.read(16384).decode('utf-8', errors='replace').splitlines()
            reasons = [line.split(':', 1)[1].strip() for line in lines
                       if line.startswith(('Claire session:', 'Claire GPU owner:', 'native session media:'))]
            reason = next((value for value in reversed(reasons) if value in allowed), 'unclassified_native_failure')
            key = state['key']
            descriptor, name = tempfile.mkstemp(prefix='native-failure-', suffix='.json', dir=self.settings.private_dir)
            with os.fdopen(descriptor, 'w') as target:
                record = {'session_id': key.session_id, 'turn_id': key.turn_id, 'epoch': key.epoch,
                          'process_exit': state['process'].returncode, 'code': reason}
                readiness = parse_claire_readiness(lines)
                if readiness is not None: record['claire_readiness'] = readiness
                json.dump(record, target)
        except OSError:
            pass

    async def close(self):
        await self.cancel()
