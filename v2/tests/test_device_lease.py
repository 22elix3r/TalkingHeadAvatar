import os
from pathlib import Path
import subprocess
import sys

import pytest

from talkinghead_v2.contracts import ContractError
from talkinghead_v2.session.device import DeviceBusy, DeviceLease


def test_competing_owner_is_denied_and_inode_survives_release(tmp_path):
    path = tmp_path / 'gpu-device-0.lock'
    first, second = DeviceLease(path), DeviceLease(path)
    first.acquire('first')
    inode = path.stat().st_ino
    try:
        with pytest.raises(DeviceBusy): second.acquire('second')
        assert 'first' not in path.read_text()
        assert path.stat().st_mode & 0o777 == 0o600
    finally:
        first.release()
    second.acquire('second')
    second.release()
    assert path.stat().st_ino == inode


def test_inherited_gpu_child_protects_device_after_parent_releases(tmp_path):
    path = tmp_path / 'gpu-device-0.lock'
    parent, competitor = DeviceLease(path), DeviceLease(path)
    parent.acquire('session')
    child = subprocess.Popen([sys.executable, '-c', 'import time;print("ready",flush=True);time.sleep(30)'],
        pass_fds=(parent.fileno,), stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        parent.release()
        with pytest.raises(DeviceBusy): competitor.acquire('new-session')
        child.terminate(); child.wait(timeout=5)
        competitor.acquire('new-session')
    finally:
        parent.release(); competitor.release()
        if child.poll() is None: child.kill(); child.wait(timeout=5)
        child.stdout.close()


@pytest.mark.parametrize('kind', ['public', 'symlink', 'hardlink', 'directory'])
def test_untrusted_lease_file_is_not_modified(tmp_path, kind):
    path = tmp_path / 'gpu-device-0.lock'
    original = tmp_path / 'unrelated'
    original.write_text('preserve')
    original.chmod(0o600)
    if kind == 'public': path.write_text('preserve'); path.chmod(0o644)
    elif kind == 'symlink': path.symlink_to(original)
    elif kind == 'hardlink': os.link(original, path)
    else: path.mkdir()
    lease = DeviceLease(path)
    with pytest.raises((OSError, ContractError)): lease.acquire('session')
    assert original.read_text() == 'preserve'
    if kind == 'public': assert path.read_text() == 'preserve'


def test_lease_requires_explicit_safe_directory_and_held_handle(tmp_path):
    with pytest.raises(ContractError): DeviceLease(Path('relative.lock'))
    lease = DeviceLease(tmp_path / 'gpu-device-0.lock')
    with pytest.raises(ContractError): _ = lease.fileno
    tmp_path.chmod(0o777)
    with pytest.raises(ContractError): lease.acquire('session')
    tmp_path.chmod(0o700)


@pytest.mark.parametrize('content', ['private settings', '{}', '[]', 'x' * 4097])
def test_existing_private_unrelated_file_is_never_truncated(tmp_path, content):
    path = tmp_path / 'gpu-device-0.lock'
    path.write_text(content); path.chmod(0o600)
    with pytest.raises(ContractError, match='preserving'):
        DeviceLease(path).acquire('session')
    assert path.read_text() == content
