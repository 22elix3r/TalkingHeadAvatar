"""Owned, offline-only Linux local profiles; no global installs or model fetches."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import select
import signal
import socket
import subprocess
import time
from urllib.parse import urlparse
from urllib.request import build_opener, ProxyHandler

PG = 'postgres@sha256:4e6e670bb069649261c9c18031f0aded7bb249a5b6664ddec29c013a89310d50'
SFU = 'livekit/livekit-server@sha256:25cd902de7cacbd5f1f779f6a6ac1eae20ba8f56b0935934ce6b0ab80d1e3d20'
NAME = re.compile(r'[a-z][a-z0-9-]{1,23}')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def private_json(path, data):
    with open(path, 'x', opener=lambda p, flags: os.open(p, flags | os.O_NOFOLLOW, 0o600)) as target:
        target.write(json.dumps(data, indent=2) + '\n')


def safe_read(path):
    info = path.lstat()
    if path.is_symlink() or not path.is_file() or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_size > 65536:
        raise ValueError('profile metadata must be a bounded private regular file owned by this user')
    return json.loads(path.read_text())


def request_json(url):
    with build_opener(ProxyHandler({})).open(url, timeout=2) as response:
        data = response.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError('local service response exceeded its bound')
    return json.loads(data)


def profile_dir(root, name):
    if not NAME.fullmatch(name):
        raise ValueError('profile name needs 2..24 lowercase letters/digits/hyphens, beginning with a letter')
    base = root / 'var/local'
    if base.is_symlink() or (base / name).is_symlink():
        raise ValueError('local profile directories cannot be symlinks')
    return base / name


def initialize(root, name, source, ports):
    from talkinghead_v2.server.config import Settings
    if len(set(ports.values())) != 5 or any(type(port) is not int or not 1024 <= port <= 65535 for port in ports.values()):
        raise ValueError('five distinct unprivileged ports are required')
    settings = Settings.load(source)
    profile = profile_dir(root, name)
    if profile.exists():
        raise ValueError('profile already exists; preserving its credentials, database and assets')
    # Share only operator-selected immutable caches, never another profile's
    # credentials, database, uploaded assets or mutable Qwen worker state.
    for path in [settings.asr_path, settings.tts_runtime, settings.web_dist]:
        if not path.exists():
            raise ValueError('selected cached model/runtime/web input is missing; no fetch attempted')
    profile.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    profile.mkdir(mode=0o700)
    password, key, secret = secrets.token_hex(24), 'tha-' + name, secrets.token_hex(32)
    config = {'database_url': f'postgresql+psycopg://tha:{password}@127.0.0.1:{ports["postgres"]}/tha_v2',
        'livekit_url': f'ws://127.0.0.1:{ports["signal"]}', 'livekit_key': key, 'livekit_secret': secret,
        'ollama_url': settings.ollama_url, 'llm_model': settings.llm_model, 'llm_digest': settings.llm_digest,
        'asr_path': str(settings.asr_path), 'asr_files': settings.asr_files,
        'tts_runtime': str(settings.tts_runtime), 'web_dist': str(settings.web_dist),
        'private_dir': str(profile), 'origin': f'http://127.0.0.1:{ports["api"]}'}
    config['media_backend'] = settings.media_backend
    if settings.native_media_runtime is not None:
        config['native_media_runtime'] = str(settings.native_media_runtime)
    private_json(profile / 'runtime.json', config)
    sfu = profile / 'livekit.yaml'
    with open(sfu, 'x', opener=lambda p, flags: os.open(p, flags | os.O_NOFOLLOW, 0o600)) as target:
        target.write(f'port: 7880\nbind_addresses: ["0.0.0.0"]\nrtc:\n  tcp_port: {ports["tcp"]}\n'
            f'  udp_port: {ports["udp"]}\n  use_external_ip: false\nkeys:\n  {key}: {secret}\nlogging:\n  level: warn\n')
    compose = {'services': {
        'postgres': {'image': PG, 'pull_policy': 'never',
            'ports': [f'127.0.0.1:{ports["postgres"]}:5432'],
            'environment': {'POSTGRES_USER': 'tha', 'POSTGRES_PASSWORD': password, 'POSTGRES_DB': 'tha_v2'},
            'volumes': ['pgdata:/var/lib/postgresql/data'],
            'healthcheck': {'test': ['CMD-SHELL', 'pg_isready -U tha -d tha_v2'], 'interval': '2s', 'timeout': '2s', 'retries': 15}},
        'livekit': {'image': SFU, 'pull_policy': 'never', 'command': ['--config', '/etc/livekit.yaml'],
            'ports': [f'127.0.0.1:{ports["signal"]}:7880', f'127.0.0.1:{ports["tcp"]}:{ports["tcp"]}', f'127.0.0.1:{ports["udp"]}:{ports["udp"]}/udp'],
            'volumes': [str(sfu) + ':/etc/livekit.yaml:ro,Z']}}, 'volumes': {'pgdata': {}}}
    private_json(profile / 'compose.json', compose)
    private_json(profile / 'profile.json', {'schema_version': 1, 'name': name, 'project': 'tha-v2-local-' + name,
        'ports': ports, 'files': {filename: digest(profile / filename) for filename in ['runtime.json', 'compose.json', 'livekit.yaml']}})
    return {'status': 'initialized', 'profile': name, 'origin': config['origin'], 'model_downloads': False}


def load_profile(root, name):
    directory = profile_dir(root, name)
    info = directory.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('profile directory must be private and owned by this user')
    marker = safe_read(directory / 'profile.json')
    if marker.get('schema_version') != 1 or marker.get('name') != name or marker.get('project') != 'tha-v2-local-' + name:
        raise ValueError('profile ownership marker is invalid')
    if set(marker.get('files', {})) != {'runtime.json', 'compose.json', 'livekit.yaml'}:
        raise ValueError('profile integrity marker is incomplete')
    for filename, expected in marker['files'].items():
        path = directory / filename
        if path.is_symlink() or path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077 or digest(path) != expected:
            raise ValueError('profile configuration changed; refusing service mutation')
    return directory, marker


def command(args, *, env=None, log=None, timeout=50):
    try:
        result = subprocess.run(args, env=env, stdin=subprocess.DEVNULL, stdout=log or subprocess.PIPE,
            stderr=log or subprocess.PIPE, timeout=timeout, check=True, text=True)
        return result.stdout or ''
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        # Never print command arguments or subprocess traces containing secrets.
        raise RuntimeError('local software/service operation failed; inspect the private profile log') from error


def compose_command(directory, marker):
    return ['docker', 'compose', '-p', marker['project'], '-f', str(directory / 'compose.json')]


def process_identity(pid):
    try:
        base = Path('/proc') / str(pid)
        data = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        if data[0] in {'Z', 'X'} or base.stat().st_uid != os.getuid():
            return None
        argv = (base / 'cmdline').read_bytes().split(b'\0')[:-1]
        if not argv:  # The kernel can clear argv between exit and zombie state.
            return None
        return {'pid': pid, 'start_ticks': int(data[19]), 'argv': argv}
    except (OSError, ValueError, IndexError):
        return None


def expected_argv(root, directory):
    return [str(directory / 'api-venv/bin/python'), '-m', 'talkinghead_v2.local_server',
        '--root', str(root), '--name', directory.name]


def owned_process(root, directory):
    record = directory / 'api-process.json'
    if not record.exists():
        return None
    saved = safe_read(record)
    live = process_identity(saved['pid'])
    if live is None:
        return None
    if live['start_ticks'] != saved['start_ticks'] or live['argv'] != [item.encode() for item in expected_argv(root, directory)]:
        raise ValueError('saved PID belongs to another process; refusing to signal or replace it')
    return live


@contextmanager
def locked(directory):
    descriptor = os.open(directory / 'operation.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def doctor(root, name):
    from talkinghead_v2.server.config import Settings
    directory, marker = load_profile(root, name)
    settings = Settings.load(directory / 'runtime.json')
    checks = {'cached_asr_hashes': all(digest(settings.asr_path / filename) == expected for filename, expected in settings.asr_files.items()),
        'tts_descriptor_present': settings.tts_runtime.is_file(), 'web_build_present': (settings.web_dist / 'index.html').is_file(),
        'offline_runtime_present': (directory / 'api-venv/bin/python').is_file()}
    if (directory / 'runtime-source.json').exists():
        try:
            source = safe_read(directory / 'runtime-source.json')
            expected = (root / 'runtimes/session-api/.venv').resolve(strict=True)
            checks['shared_runtime_identity'] = (source['mode'] == 'shared_qualified_v2' and
                (directory / 'api-venv').is_symlink() and (directory / 'api-venv').resolve(strict=True) == expected and
                source['source'] == str(expected) and source['lock_sha256'] == digest(root / 'runtimes/session-api/uv.lock') and
                source['fingerprint'] == runtime_fingerprint(expected))
        except (OSError, ValueError, KeyError, RuntimeError):
            checks['shared_runtime_identity'] = False
    for label, image in [('postgres_image', PG), ('livekit_image', SFU)]:
        try:
            command(['docker', 'image', 'inspect', '--format={{.Id}}', image], timeout=5)
            checks[label] = True
        except RuntimeError:
            checks[label] = False
    try:
        tags = request_json(settings.ollama_url + '/api/tags')
        checks['cached_llm_digest'] = any(model['name'] == settings.llm_model and model['digest'] == settings.llm_digest for model in tags['models'])
    except (OSError, ValueError, KeyError):
        checks['cached_llm_digest'] = False
    if settings.media_backend == 'native_static':
        from talkinghead_v2.server.native_media import NativeMediaConfig
        try:
            NativeMediaConfig.load(settings.native_media_runtime)
            checks['pinned_native_media_runtime'] = True
        except (OSError, ValueError, KeyError):
            checks['pinned_native_media_runtime'] = False
    elif settings.media_backend == 'native_claire_research':
        from talkinghead_v2.server.claire_media import ClaireMediaConfig
        try:
            config = ClaireMediaConfig.load(settings.native_media_runtime)
            config.validate_device()
            checks['pinned_claire_media_runtime'] = True
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            checks['pinned_claire_media_runtime'] = False
    return {'status': 'ready' if all(checks.values()) else 'incomplete', 'profile': name, 'checks': checks,
        'media_backend': settings.media_backend, 'no_fetch_attempted': True}


def runtime_fingerprint(source):
    # Read-only audit of already-installed wheel RECORDs. This never imports
    # inference packages, downloads artifacts or changes the source environment.
    script = r'''
import base64,hashlib,importlib.metadata as metadata,json,pathlib,sys
prefix=pathlib.Path(sys.prefix).resolve()
packages=[]
for distribution in metadata.distributions():
    files=[]
    for entry in distribution.files or ():
        expected=entry.hash
        if expected is None:continue
        if expected.mode!='sha256':raise ValueError('unsupported installed RECORD hash')
        path=distribution.locate_file(entry).resolve()
        if not path.is_relative_to(prefix) or not path.is_file():raise ValueError('installed file outside qualified runtime')
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        if base64.urlsafe_b64encode(digest.digest()).decode().rstrip('=')!=expected.value:raise ValueError('installed RECORD mismatch')
        files.append((str(entry),digest.hexdigest()))
    if not files:raise ValueError('installed distribution lacks a verifiable RECORD')
    packages.append((distribution.metadata['Name'].lower().replace('_','-'),distribution.version,
        hashlib.sha256(json.dumps(sorted(files),separators=(',',':')).encode()).hexdigest()))
print(json.dumps({'python':sys.version.split()[0],'packages':sorted(packages)}))
'''
    result = command([str(source / 'bin/python'), '-I', '-c', script], timeout=30)
    data = json.loads(result)
    if not data['python'].startswith('3.12.') or not data['packages']:
        raise ValueError('qualified V2 Python 3.12 runtime required')
    return data


def provision(root, name, *, borrow_runtime=None):
    directory, _ = load_profile(root, name)
    with locked(directory):
        if owned_process(root, directory):
            raise ValueError('stop the owned API before provisioning its software environment')
        if borrow_runtime is not None:
            source = borrow_runtime.resolve(strict=True)
            if source != (root / 'runtimes/session-api/.venv').resolve(strict=True) or not source.is_relative_to(root.resolve()):
                raise ValueError('only this checkout\'s qualified V2 API runtime may be shared')
            fingerprint = runtime_fingerprint(source)
            env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(source), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', UV_OFFLINE='1')
            audit = subprocess.run(['uv', 'sync', '--locked', '--offline', '--dry-run', '--project', str(root / 'runtimes/session-api'),
                '--python', '/usr/bin/python3.12'], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, timeout=30, check=True)
            if 'Would make no changes' not in audit.stdout:
                raise ValueError('source runtime does not match the existing dependency lock')
            target = directory / 'api-venv'
            if target.is_symlink():
                if target.resolve() != source: raise ValueError('refusing to replace another runtime link')
            else:
                if target.exists(): target.rename(directory / ('api-venv-retained-' + str(time.time_ns())))
                target.symlink_to(source, target_is_directory=True)
            record = directory / 'runtime-source.json'
            data = {'mode': 'shared_qualified_v2', 'source': str(source), 'lock_sha256': digest(root / 'runtimes/session-api/uv.lock'),
                    'fingerprint': fingerprint, 'downloads': False}
            if record.exists():
                if safe_read(record) != data: raise ValueError('shared runtime identity changed; retain the existing record for review')
            else: private_json(record, data)
            return doctor(root, name)
        env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(directory / 'api-venv'),
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', UV_OFFLINE='1')
        with open(directory / 'operator.log', 'a', opener=lambda p, flags: os.open(p, flags | os.O_NOFOLLOW, 0o600)) as log:
            command(['uv', 'sync', '--locked', '--offline', '--project', str(root / 'runtimes/session-api'),
                '--python', '/usr/bin/python3.12'], env=env, log=log, timeout=120)
    return doctor(root, name)


def up(root, name):
    from talkinghead_v2.server.config import Settings
    directory, marker = load_profile(root, name)
    settings = Settings.load(directory / 'runtime.json')
    with locked(directory):
        live = owned_process(root, directory)
        if live:
            ready = request_json(settings.origin + '/health/ready')
            return {'status': 'running', 'profile': name, 'origin': settings.origin, 'ready': ready}
        if doctor(root, name)['status'] != 'ready':
            raise ValueError('offline profile is incomplete; run local doctor/provision first')
        with socket.socket() as probe:
            # Match the server's restart semantics: closed connections in
            # TIME_WAIT are not an active listener and must not block restart.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', marker['ports']['api']))
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', THA_P04_CONFIG=str(directory / 'runtime.json'))
        with open(directory / 'operator.log', 'a', opener=lambda p, flags: os.open(p, flags | os.O_NOFOLLOW, 0o600)) as log:
            command(compose_command(directory, marker) + ['up', '-d', '--wait', '--wait-timeout', '40', '--pull', 'never'], log=log)
            command([str(directory / 'api-venv/bin/alembic'), '-c', str(root / 'runtimes/session-api/alembic.ini'), 'upgrade', 'head'], env=env, log=log)
            child = subprocess.Popen(expected_argv(root, directory), env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, start_new_session=True)
            # The child records its own identity under a lifetime lock before
            # loading models. A launcher crash cannot leave an unrecorded API.
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError('local API startup failed; inspect the private profile log')
                try:
                    ready = request_json(settings.origin + '/health/ready')
                    return {'status': 'running', 'profile': name, 'origin': settings.origin, 'ready': ready}
                except (OSError, ValueError):
                    time.sleep(.2)
            raise TimeoutError('local API readiness deadline exceeded; process ownership retained for local down')


def down(root, name):
    directory, marker = load_profile(root, name)
    with locked(directory):
        live = owned_process(root, directory)
        if live:
            # A pidfd pins the actual kernel process across exit/PID reuse.
            # Revalidate ownership after opening it, before sending any signal.
            handle = os.pidfd_open(live['pid'])
            try:
                confirmed = owned_process(root, directory)
                if confirmed != live:
                    raise ValueError('API process changed before shutdown; no signal sent')
                signal.pidfd_send_signal(handle, signal.SIGTERM)
                exited = select.poll()
                exited.register(handle, select.POLLIN)
                if not exited.poll(20000):
                    # Keep its dependencies and ownership record available.
                    # Killing only the API could strand a GPU worker child.
                    raise RuntimeError('owned API drain timed out; services and process record retained for recovery')
            finally:
                os.close(handle)
        with open(directory / 'operator.log', 'a', opener=lambda p, flags: os.open(p, flags | os.O_NOFOLLOW, 0o600)) as log:
            command(compose_command(directory, marker) + ['down', '--timeout', '15'], log=log)
        (directory / 'api-process.json').unlink(missing_ok=True)
    return {'status': 'stopped', 'profile': name, 'database_volume_retained': True, 'private_assets_retained': True}
