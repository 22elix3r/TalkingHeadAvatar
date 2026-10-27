"""Profile isolation and shutdown ownership; no Docker, models or GPU required."""
import subprocess
import sys

import pytest

from talkinghead_v2 import local


def source(tmp_path):
    for name in ['asr', 'web']:
        (tmp_path / name).mkdir()
    (tmp_path / 'web/index.html').write_text('local fixture')
    (tmp_path / 'asr/config.json').write_text('{}')
    (tmp_path / 'tts.json').write_text('{}')
    path = tmp_path / 'source.json'
    local.private_json(path, {'database_url': 'postgresql+psycopg://old:old@127.0.0.1:15432/old',
        'livekit_url': 'ws://127.0.0.1:17880', 'livekit_key': 'old', 'livekit_secret': 'old',
        'ollama_url': 'http://127.0.0.1:11434', 'llm_model': 'cached', 'llm_digest': 'pinned',
        'asr_path': 'asr', 'asr_files': {'config.json': local.digest(tmp_path / 'asr/config.json')},
        'tts_runtime': 'tts.json', 'web_dist': 'web', 'private_dir': '.', 'origin': 'http://127.0.0.1:18700'})
    return path


PORTS = dict(api=18710, postgres=15442, signal=17890, tcp=17891, udp=17892)


def initialized(tmp_path):
    path = source(tmp_path)
    local.initialize(tmp_path, 'check', path, PORTS)
    return local.load_profile(tmp_path, 'check')


def test_isolated_credentials_and_retained_original(tmp_path):
    original = source(tmp_path)
    before = original.read_bytes()
    local.initialize(tmp_path, 'check', original, PORTS)
    directory, marker = local.load_profile(tmp_path, 'check')
    config = local.safe_read(directory / 'runtime.json')
    compose = local.safe_read(directory / 'compose.json')
    assert original.read_bytes() == before
    assert config['livekit_secret'] != 'old'
    assert ':15442/tha_v2' in config['database_url']
    assert config['private_dir'] == str(directory)
    assert marker['project'] == 'tha-v2-local-check'
    assert all(service['pull_policy'] == 'never' for service in compose['services'].values())
    assert all(port.startswith('127.0.0.1:') for service in compose['services'].values() for port in service['ports'])
    assert (directory.stat().st_mode & 0o777) == 0o700
    assert all((directory / name).stat().st_mode & 0o777 == 0o600 for name in marker['files'])
    with pytest.raises(ValueError, match='already exists'):
        local.initialize(tmp_path, 'check', original, PORTS)


@pytest.mark.parametrize('name', ['../other', '/tmp/test', 'UPPER', 'a', 'a' * 25, 'a\nb'])
def test_profile_name_cannot_escape_owned_root(tmp_path, name):
    with pytest.raises(ValueError):
        local.profile_dir(tmp_path, name)


def test_duplicate_ports_and_missing_caches_do_not_create_profile(tmp_path):
    path = source(tmp_path)
    with pytest.raises(ValueError, match='distinct'):
        local.initialize(tmp_path, 'check', path, dict(PORTS, tcp=PORTS['udp']))
    (tmp_path / 'tts.json').unlink()
    with pytest.raises(ValueError, match='missing'):
        local.initialize(tmp_path, 'check', path, PORTS)
    assert not (tmp_path / 'var').exists()


def test_tampered_or_public_profile_is_not_used_for_mutations(tmp_path):
    directory, _ = initialized(tmp_path)
    path = directory / 'compose.json'
    old = path.read_bytes()
    path.write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        local.load_profile(tmp_path, 'check')
    path.write_bytes(old)
    path.chmod(0o644)
    with pytest.raises(ValueError):
        local.load_profile(tmp_path, 'check')


def test_profile_symlink_and_operation_overlap_are_rejected(tmp_path):
    directory, _ = initialized(tmp_path)
    with local.locked(directory):
        with pytest.raises(BlockingIOError):
            with local.locked(directory):
                pytest.fail('overlapping mutation acquired the lock')
    (directory.parent / 'alias').symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match='symlinks'):
        local.load_profile(tmp_path, 'alias')


def test_foreign_or_reused_pid_is_never_signaled(tmp_path, monkeypatch):
    directory, _ = initialized(tmp_path)
    process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
    try:
        live = local.process_identity(process.pid)
        local.private_json(directory / 'api-process.json', {'pid': process.pid, 'start_ticks': live['start_ticks']})
        signals = []
        monkeypatch.setattr(local.signal, 'pidfd_send_signal', lambda *args: signals.append(args))
        with pytest.raises(ValueError, match='another process'):
            local.down(tmp_path, 'check')
        assert signals == [] and process.poll() is None
        saved = directory / 'api-process.json'
        saved.unlink()
        local.private_json(saved, {'pid': process.pid, 'start_ticks': live['start_ticks'] - 1})
        with pytest.raises(ValueError, match='another process'):
            local.down(tmp_path, 'check')
        assert signals == [] and process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_offline_provision_pins_project_and_never_fetches(tmp_path, monkeypatch):
    directory, _ = initialized(tmp_path)
    calls = []
    monkeypatch.setattr(local, 'command', lambda argv, **kwargs: calls.append((argv, kwargs)))
    monkeypatch.setattr(local, 'doctor', lambda *args: {'status': 'ready'})
    assert local.provision(tmp_path, 'check')['status'] == 'ready'
    argv, options = calls[0]
    assert '--offline' in argv and '--locked' in argv
    assert options['env']['UV_PROJECT_ENVIRONMENT'] == str(directory / 'api-venv')
    assert options['env']['HF_HUB_OFFLINE'] == options['env']['TRANSFORMERS_OFFLINE'] == '1'
    assert not any('pip' in part or 'pull' in part for part in argv)


def test_shutdown_waits_on_owned_kernel_process_before_stopping_services(tmp_path, monkeypatch):
    directory, _ = initialized(tmp_path)
    argv = [sys.executable, '-c', 'import time;time.sleep(30)']
    process = subprocess.Popen(argv)
    try:
        live = local.process_identity(process.pid)
        local.private_json(directory / 'api-process.json', {'pid': process.pid, 'start_ticks': live['start_ticks']})
        monkeypatch.setattr(local, 'expected_argv', lambda *args: argv)
        commands = []
        def services(command, **kwargs):
            assert process.poll() is not None
            commands.append(command)
        monkeypatch.setattr(local, 'command', services)
        result = local.down(tmp_path, 'check')
        assert result['database_volume_retained'] and result['private_assets_retained']
        assert commands[0][-3:] == ['down', '--timeout', '15']
        assert '--volumes' not in commands[0]
        assert not (directory / 'api-process.json').exists()
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)


def test_drain_timeout_retains_dependencies_and_ownership(tmp_path, monkeypatch):
    directory, _ = initialized(tmp_path)
    argv = [sys.executable, '-c',
        'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print("ready",flush=True);time.sleep(30)']
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'ready'
        live = local.process_identity(process.pid)
        local.private_json(directory / 'api-process.json', {'pid': process.pid, 'start_ticks': live['start_ticks']})
        monkeypatch.setattr(local, 'expected_argv', lambda *args: argv)
        class Deadline:
            def register(self, *args): pass
            def poll(self, timeout):
                assert timeout == 20000
                return []
        monkeypatch.setattr(local.select, 'poll', Deadline)
        commands = []
        monkeypatch.setattr(local, 'command', lambda *args, **kwargs: commands.append(args))
        with pytest.raises(RuntimeError, match='drain timed out'):
            local.down(tmp_path, 'check')
        assert process.poll() is None
        assert (directory / 'api-process.json').exists()
        assert commands == []
    finally:
        process.kill()
        process.wait(timeout=5)
        process.stdout.close()
