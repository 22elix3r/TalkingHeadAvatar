"""Build an existing pinned/patched SDK offline; never fetch sources or weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from talkinghead_v2.server.inbound_config import CARGO_LOCK_SHA256, PATCH_SHA256, RUST_REVISION

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def output(command, source):
    return subprocess.check_output(command, cwd=source, text=True).strip()


def verify_tree(source, revision, expected_delta=b''):
    delta = subprocess.check_output(['git', 'diff', '--no-ext-diff', '--no-textconv', 'HEAD', '--'], cwd=source)
    untracked = output(['git', 'ls-files', '--others', '--exclude-standard'], source)
    if output(['git', 'rev-parse', 'HEAD'], source) != revision or delta != expected_delta or untracked:
        raise ValueError('pinned complete source tree required; reject staged, untracked or submodule changes')


def verify_sources(source, patch, names):
    verify_tree(source, RUST_REVISION, patch.read_bytes())
    if digest(patch) != PATCH_SHA256 or digest(source / 'Cargo.lock') != CARGO_LOCK_SHA256:
        raise ValueError('pinned SDK source, lock and exact patch required')
    for name, revision in [('livekit-protocol/protocol', '9a7c5cf96bbbcbc30a8ce76bca16815ba5b7bb33'),
                           ('yuv-sys/libyuv', '917276084a49be726c90292ff0a6b0a3d571a6af')]:
        verify_tree(source / name, revision)
    return {name: digest(source / name) for name in names}


def main(args):
    source, target, cargo_home = (path.resolve(strict=True) for path in (args.source, args.target, args.cargo_home))
    if args.report.exists():
        raise ValueError('preserve previous build evidence')
    patch = ROOT / 'patches/livekit-native-audio-fail-closed.patch'
    names = ['libwebrtc/src/audio_stream.rs', 'libwebrtc/src/native/audio_stream.rs',
             'livekit-ffi/protocol/audio_frame.proto', 'livekit-ffi/src/server/audio_stream.rs']
    source_hashes = verify_sources(source, patch, names)
    tools = {name: args.rust_sysroot.resolve() / 'bin' / name for name in ('cargo', 'rustc', 'rustdoc')}
    if not all(path.is_file() for path in tools.values()) or not output([str(tools['rustc']), '--version'], source).startswith('rustc 1.99.0 '):
        raise ValueError('existing qualified Rust 1.99.0 required; no automatic install')
    webrtc, protoc = args.webrtc.resolve(strict=True), args.protoc.resolve(strict=True)
    if not (webrtc / 'lib/libwebrtc.a').is_file() or not protoc.is_file():
        raise ValueError('existing WebRTC and protoc required')
    environment = dict(os.environ, CARGO_HOME=str(cargo_home), CARGO_TARGET_DIR=str(target),
        LK_CUSTOM_WEBRTC=str(webrtc), PROTOC=str(protoc), CARGO_NET_OFFLINE='true', CARGO_BUILD_JOBS='4',
        CARGO_INCREMENTAL='0', RUSTC=str(tools['rustc']), RUSTDOC=str(tools['rustdoc']))
    args.report.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    logs = []
    commands = [('native-audio-tests', ['test', '--locked', '--offline', '-p', 'libwebrtc', '--release', 'imp::audio_stream::tests', '--', '--nocapture']),
                ('ffi-build', ['build', '--locked', '--offline', '-p', 'livekit-ffi', '--release'])]
    for name, command in commands:
        if shutil.disk_usage(target).free < 3 * 1024 ** 3:
            raise ValueError('build disk reserve reached')
        log = args.report.parent / (args.report.stem + '-' + name + '.log')
        with open(log, 'xb', opener=lambda path, flags: os.open(path, flags | os.O_NOFOLLOW, 0o600)) as destination:
            result = subprocess.run([str(tools['cargo']), *command], cwd=source, env=environment,
                stdin=subprocess.DEVNULL, stdout=destination, stderr=subprocess.STDOUT, timeout=900)
        if result.returncode:
            raise ValueError('offline build failed; inspect its private log')
        logs.append(log)
    if '4 passed; 0 failed' not in logs[0].read_text():
        raise ValueError('native audio test selection did not run four passing cases')
    if source_hashes != verify_sources(source, patch, names):
        raise ValueError('source changed during build')
    library = target / 'release/liblivekit_ffi.so'
    report = {'schema_version': 1, 'rust_revision': RUST_REVISION, 'patch_sha256': PATCH_SHA256,
        'cargo_lock_sha256': CARGO_LOCK_SHA256, 'library_sha256': digest(library), 'native_audio_tests_passed': 4,
        'rust_version': output([str(tools['rustc']), '--version'], source), 'protoc_sha256': digest(protoc),
        'webrtc_sha256': digest(webrtc / 'lib/libwebrtc.a'), 'source_sha256': source_hashes,
        'build_script_sha256': digest(Path(__file__)), 'logs_sha256': {path.name: digest(path) for path in logs},
        'offline': True, 'model_downloads': False}
    with open(args.report, 'x', opener=lambda path, flags: os.open(path, flags | os.O_NOFOLLOW, 0o600)) as destination:
        destination.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'built': True, 'native_audio_tests_passed': 4, 'library_sha256': report['library_sha256'], 'offline': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'target', 'cargo-home', 'webrtc', 'protoc', 'rust-sysroot', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
