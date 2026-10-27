"""Pin an already-built first-party native runtime; never build or fetch inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from talkinghead_v2.server.native_media import NativeMediaConfig, SDK_REVISION, private_write

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build = ROOT / 'var/build/native-media'
    libraries = ROOT / 'var/build/livekit-cpp/lib'
    paths = {'executable': build / 'tha_session_media', 'ptx': build / 'media_kernels.ptx',
             'livekit_library': libraries / 'liblivekit.so', 'ffi_library': libraries / 'liblivekit_ffi.so'}
    payload = {'schema_version': 1, 'sdk_revision': SDK_REVISION, 'prebuffer_ms': 300,
               **{name: {'path': str(path.resolve(strict=True)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                  for name, path in paths.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    private_write(args.output, (json.dumps(payload, indent=2) + '\n').encode())
    NativeMediaConfig.load(args.output)
    print(json.dumps({'pinned': True, 'models_downloaded': False, 'prebuffer_ms': 300}))


if __name__ == '__main__': main()
