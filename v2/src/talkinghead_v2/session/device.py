"""Fail-closed single-device research admission; no CUDA/model initialization."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import time

from talkinghead_v2.contracts import ContractError


class DeviceBusy(ContractError):
    """Another serving/qualification worker owns the configured device lease."""


class DeviceLease:
    """One shared open-file description can be inherited by every GPU child.

    Never unlink the lock file: replacing its inode would admit two owners.
    Kernel flock ownership, rather than persisted PID metadata, is authoritative.
    """
    def __init__(self, path: Path):
        if not path.is_absolute():
            raise ContractError('device lease path must be explicit and absolute')
        self.path = path
        self._fd = None

    @property
    def fileno(self):
        if self._fd is None:
            raise ContractError('device lease is not held')
        return self._fd

    def acquire(self, owner: str):
        if self._fd is not None:
            return
        parent = self.path.parent.stat()
        if parent.st_uid != os.getuid() or parent.st_mode & 0o022:
            raise ContractError('device lease directory must be owned and not writable by other users')
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
                raise ContractError('device lease must be a private singly-linked regular file')
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise DeviceBusy('configured GPU device is already admitted') from error
            existing = os.fstat(descriptor).st_size
            if existing:
                try:
                    if existing > 4096: raise ValueError('oversized lease metadata')
                    old = json.loads(os.pread(descriptor, existing, 0))
                    if not isinstance(old, dict) or old.get('schema_version') != 1 or old.get('scope') != 'cuda_device_0_research_single_session':
                        raise ValueError('not an owned lease marker')
                except (ValueError, UnicodeError) as error:
                    raise ContractError('existing device lease path contains unrelated or corrupt data; preserving it') from error
            metadata = {'schema_version': 1, 'scope': 'cuda_device_0_research_single_session', 'pid': os.getpid(),
                'owner_sha256': hashlib.sha256(owner.encode()).hexdigest(), 'acquired_monotonic_ns': time.monotonic_ns()}
            os.ftruncate(descriptor, 0)
            os.write(descriptor, (json.dumps(metadata) + '\n').encode())
            self._fd = descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def release(self):
        # Do not LOCK_UN: inherited GPU workers share this description. Closing
        # just the parent reference must keep admission closed while a child lives.
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
