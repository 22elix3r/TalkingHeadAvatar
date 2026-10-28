"""Real temporary Git trees exercise staged/untracked provenance boundaries."""
import importlib.util
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/build_inbound_audio_ffi.py'
spec = importlib.util.spec_from_file_location('inbound_build', SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


@pytest.fixture
def source(tmp_path):
    git(tmp_path, 'init', '-q')
    (tmp_path / 'source.rs').write_text('original\n')
    (tmp_path / 'other.rs').write_text('original\n')
    git(tmp_path, 'add', '.')
    git(tmp_path, '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid',
        '-c', 'commit.gpgsign=false', 'commit', '-qm', 'local test fixture')
    revision = git(tmp_path, 'rev-parse', 'HEAD')
    (tmp_path / 'source.rs').write_text('qualified change\n')
    delta = subprocess.check_output(['git', '-C', str(tmp_path), 'diff', 'HEAD', '--'])
    return tmp_path, revision, delta


def test_exact_patch_can_be_staged_without_changing_identity(source):
    path, revision, delta = source
    builder.verify_tree(path, revision, delta)
    git(path, 'add', 'source.rs')
    builder.verify_tree(path, revision, delta)


@pytest.mark.parametrize('change', ['staged_extra', 'untracked', 'clean_submodule_required'])
def test_extra_source_changes_are_rejected(source, change):
    path, revision, delta = source
    if change == 'staged_extra':
        (path / 'other.rs').write_text('unqualified staged code\n')
        git(path, 'add', 'other.rs')
    elif change == 'untracked':
        (path / 'unqualified.rs').write_text('untracked source\n')
    else:
        # Submodule checks use this same guard with no permitted patch.
        delta = b''
    with pytest.raises(ValueError, match='complete source tree'):
        builder.verify_tree(path, revision, delta)
