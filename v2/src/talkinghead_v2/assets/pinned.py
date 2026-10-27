"""Explicit immutable artifact provisioning; verification never performs network I/O."""
from dataclasses import dataclass
import fcntl
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen

from talkinghead_v2.contracts import ContractError
from talkinghead_v2.assets.artifacts import sha256_file


@dataclass(frozen=True)
class PinnedFile:
    path: str
    size: int
    sha256: str
    url: str

    def __post_init__(self):
        name = PurePosixPath(self.path)
        if (self.path in {'', '.'} or name.is_absolute() or name.as_posix() != self.path or '..' in name.parts or
            any(value in self.path for value in ('\\', '\0', '%', '?', '#')) or
            type(self.size) is not int or not 0 <= self.size <= 1024**3 or
            not re.fullmatch(r'[0-9a-f]{64}', self.sha256)):
            raise ContractError('unsafe or unpinned artifact entry')
        url = urlparse(self.url)
        if url.scheme != 'https' or not url.hostname or url.username or url.password or url.fragment:
            raise ContractError('artifact source requires HTTPS without embedded credentials')


def check(path, entry):
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != entry.size or
        sha256_file(path) != entry.sha256):
        raise ContractError('artifact missing, unsafe, wrong size or hash: ' + entry.path)


def verify(destination, entries):
    if destination.is_symlink() or not destination.is_dir(): raise ContractError('verified snapshot absent')
    expected = {entry.path for entry in entries}
    actual = set()
    for path in destination.rglob('*'):
        if path.is_symlink() or not (path.is_file() or path.is_dir()): raise ContractError('unsafe artifact snapshot entry')
        if path.is_file(): actual.add(path.relative_to(destination).as_posix())
    if actual != expected: raise ContractError('snapshot inventory mismatch')
    for entry in entries: check(destination / entry.path, entry)
    return destination


def provision(cache, name, entries, *, fetch=None, reserve_bytes=4*1024**3):
    """Caller supplies a source-specific allowlist; no discovery or remote code execution."""
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', name) or not entries or len({entry.path for entry in entries}) != len(entries):
        raise ContractError('bounded unique artifact snapshot required')
    if fetch is None:
        if os.environ.get('HF_HUB_OFFLINE') == '1' or os.environ.get('TRANSFORMERS_OFFLINE') == '1':
            raise ContractError('offline mode forbids explicit network provisioning')
        fetch = lambda url: urlopen(url, timeout=40)
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    cache = cache.resolve()
    parent = cache.stat()
    if parent.st_uid != os.getuid() or parent.st_mode & 0o022: raise ContractError('private operator artifact cache required')
    lock_fd = os.open(cache / '.provision.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        destination = cache / name
        if destination.exists() or destination.is_symlink(): return verify(destination, entries)
        blobs = cache / 'blobs'
        if blobs.is_symlink(): raise ContractError('artifact blob directory cannot be a symlink')
        blobs.mkdir(exist_ok=True, mode=0o700)
        needed = sum(entry.size for entry in entries if not (blobs / entry.sha256).exists())
        if shutil.disk_usage(cache).free < needed + reserve_bytes: raise ContractError('artifact download would exhaust the disk reserve')
        for entry in entries:
            blob = blobs / entry.sha256
            if blob.exists() or blob.is_symlink(): check(blob, entry); continue
            download_fd, temporary_name = tempfile.mkstemp(prefix='.download-', dir=blobs)
            temporary = Path(temporary_name)
            try:
                digest = hashlib.sha256(); count = 0
                with os.fdopen(download_fd, 'wb') as target, fetch(entry.url) as source:
                    while block := source.read(min(1024 * 1024, entry.size - count + 1)):
                        count += len(block)
                        if count > entry.size: raise ContractError('artifact response exceeds declared size')
                        digest.update(block); target.write(block)
                    target.flush(); os.fsync(target.fileno())
                if count != entry.size or digest.hexdigest() != entry.sha256: raise ContractError('artifact response size/hash mismatch')
                temporary.chmod(0o444); temporary.rename(blob)
            finally: temporary.unlink(missing_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.snapshot-', dir=cache))
        try:
            for entry in entries:
                target = staging / entry.path
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                os.link(blobs / entry.sha256, target)
            staging.rename(destination)
        finally:
            if staging.exists(): shutil.rmtree(staging)
        return verify(destination, entries)
    finally: os.close(lock_fd)
