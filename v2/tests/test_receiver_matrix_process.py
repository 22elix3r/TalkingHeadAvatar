"""A timed-out qualification cannot strand its own decoder/GPU descendants."""
import fcntl
import importlib.util
import os
from pathlib import Path
import sys
import time
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/qualify_claire_receiver_matrix.py'
spec = importlib.util.spec_from_file_location('receiver_matrix', SCRIPT)
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


@pytest.mark.parametrize('ending', ['timeout', 'exit3', 'sigterm', 'exit0'])
def test_failure_fences_descendants_that_ignore_sigterm(tmp_path,ending):
    lock = tmp_path / 'owner.lock'
    ready = tmp_path / 'ready'
    child = tmp_path / 'child.py'
    child.write_text('import fcntl,signal,time\nfrom pathlib import Path\n'
        'signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
        f'owner=open({str(lock)!r},"w")\nfcntl.flock(owner,fcntl.LOCK_EX)\n'
        f'Path({str(ready)!r}).write_text("owned")\ntime.sleep(60)\n')
    parent = tmp_path / 'parent.py'
    ending_code={'timeout':'time.sleep(60)','exit3':'raise SystemExit(3)',
                 'sigterm':'os.kill(os.getpid(),signal.SIGTERM)','exit0':'raise SystemExit(0)'}[ending]
    parent.write_text('import subprocess,sys,time,os,signal\nfrom pathlib import Path\n'
        f'subprocess.Popen([sys.executable,{str(child)!r}])\n'
        f'while not Path({str(ready)!r}).exists():time.sleep(.01)\n'
        +ending_code+'\n')
    with (tmp_path / 'process.log').open('wb') as log:
        code = matrix.run_owned_process([sys.executable, str(parent)], dict(os.environ), log, timeout=1, grace=.1)
    expected={'timeout':'timeout','exit3':3,'sigterm':-15,'exit0':'descendants_after_success'}[ending]
    assert code == expected and ready.read_text() == 'owned'
    with lock.open('r') as owner:
        # SIGKILL delivery and descendant teardown are asynchronous with the
        # leader's wait. Require actual release, rather than instant release.
        deadline = time.monotonic() + 2
        while True:
            try:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline: raise
                time.sleep(.01)


def test_normal_qualification_returns_its_actual_exit_code(tmp_path):
    with (tmp_path / 'process.log').open('wb') as log:
        assert matrix.run_owned_process([sys.executable, '-c', 'raise SystemExit(3)'], dict(os.environ), log) == 3
