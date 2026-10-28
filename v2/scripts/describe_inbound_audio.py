"""Pin an already-built and process-qualified inbound audio runtime; no fetch."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil

from talkinghead_v2.server.inbound_config import (
    InboundAudioConfig, PATCH_SHA256, PYTHON_LIVEKIT_VERSION, RUST_REVISION,
    file_hash, private_json,
)

ROOT = Path(__file__).resolve().parents[1]


def write_private(path, value, mode=0o600):
    with open(path, 'xb', opener=lambda name, flags: os.open(name, flags | os.O_NOFOLLOW, mode)) as target:
        target.write(value)


def main(args):
    if args.output.exists():
        raise ValueError('preserve an existing inbound runtime descriptor')
    build, matrix = private_json(args.build), private_json(args.matrix)
    library = args.library.resolve(strict=True)
    digest = file_hash(library)
    if (version('livekit') != PYTHON_LIVEKIT_VERSION or build.get('library_sha256') != digest or
        matrix.get('ffi_sha256', {}).get('patched') != digest or matrix.get('status') != 'passed' or
        matrix.get('probe_sha256') != file_hash(ROOT / 'scripts/qualify_native_audio_queue.py') or
        matrix.get('runner_sha256') != file_hash(ROOT / 'scripts/qualify_inbound_audio_matrix.py') or
        file_hash(ROOT / 'patches/livekit-native-audio-fail-closed.patch') != PATCH_SHA256):
        raise ValueError('current binary, source, matrix and Python SDK qualification required')
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    snapshot = args.output.parent / ('inbound-audio-' + digest[:16])
    snapshot.mkdir(mode=0o700)
    target = snapshot / 'library.so'
    with library.open('rb') as source, open(target, 'xb', opener=lambda name, flags: os.open(name, flags | os.O_NOFOLLOW, 0o500)) as destination:
        shutil.copyfileobj(source, destination, 1024 * 1024)
    paths = {'library': target, 'build': snapshot / 'build.json', 'matrix': snapshot / 'matrix.json'}
    write_private(paths['build'], args.build.read_bytes())
    write_private(paths['matrix'], args.matrix.read_bytes())
    for role, case in [('normal', 'normal-0'), ('overflow', 'overflow-0'), ('control', 'eviction-control')]:
        observed = next(row['observation'] for row in matrix['cases'] if row['name'] == case)
        paths[role] = snapshot / (role + '.json')
        write_private(paths[role], (json.dumps(observed, indent=2) + '\n').encode())
    descriptor = {'schema_version': 1, 'policy': 'native_audio_overflow_ends_stream', 'rust_revision': RUST_REVISION,
        'patch_sha256': PATCH_SHA256, 'python_livekit_version': PYTHON_LIVEKIT_VERSION, 'native_frames': 25, 'python_frames': 12,
        'files': {role: {'path': str(path.relative_to(args.output.parent)), 'size': path.stat().st_size,
                         'sha256': file_hash(path)} for role, path in paths.items()}}
    write_private(args.output, (json.dumps(descriptor, indent=2) + '\n').encode())
    config = InboundAudioConfig.load(args.output)
    print(json.dumps({'pinned': True, 'revision': config.revision, 'native_queue_ms': 250,
                      'python_queue_ms': 240, 'model_downloads': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'build', 'matrix', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
