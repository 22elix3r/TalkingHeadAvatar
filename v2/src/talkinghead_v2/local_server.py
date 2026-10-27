"""Owned local API entrypoint, imported only inside the separate API environment."""
import argparse
import fcntl
import os
from pathlib import Path
from urllib.parse import urlparse

from talkinghead_v2.local import load_profile, private_json, process_identity, safe_read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    directory, _ = load_profile(args.root.resolve(), args.name)
    guard = os.open(directory / 'api-owner.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = process_identity(os.getpid())
        record = {'pid': identity['pid'], 'start_ticks': identity['start_ticks']}
        temporary = directory / f'api-process.{os.getpid()}.json'
        private_json(temporary, record)
        os.replace(temporary, directory / 'api-process.json')
        try:
            import uvicorn
            from talkinghead_v2.server.app import create_app
            from talkinghead_v2.server.config import Settings
            settings = Settings.load(directory / 'runtime.json')
            origin = urlparse(settings.origin)
            if origin.scheme != 'http' or origin.hostname != '127.0.0.1' or origin.port is None or origin.path or origin.query or origin.fragment or origin.username:
                raise ValueError('local serving requires an explicit http://127.0.0.1:port origin')
            uvicorn.run(create_app(settings), host='127.0.0.1', port=origin.port, access_log=False, log_level='warning')
        finally:
            path = directory / 'api-process.json'
            if path.exists() and safe_read(path) == record:
                path.unlink()
    finally:
        os.close(guard)


if __name__ == '__main__':
    main()
