"""Read-only/no-spawn competitor check while a real session owns cached Qwen."""
import argparse
import asyncio
import json
from pathlib import Path
import subprocess

from talkinghead_v2.contracts import GenerationKey
from talkinghead_v2.providers.tts.native import QwenNativeProvider, load_native_config
from talkinghead_v2.session.device import DeviceBusy


async def run(args):
    config = load_native_config(args.runtime)
    holder_text = subprocess.check_output(['fuser', str(config.device_lease_path)], text=True,
        stderr=subprocess.PIPE, timeout=5)
    native_holders = 0
    for value in holder_text.split():
        argv = (Path('/proc') / value / 'cmdline').read_bytes().split(b'\0')
        native_holders += argv[0] == config.command[0].encode()
    if native_holders != 1:
        raise ValueError('exactly one actual native Qwen worker must inherit the device lease')
    candidate = QwenNativeProvider(config, diagnostic_log=args.output.with_suffix('.private.log'))
    attempted = False
    original = asyncio.create_subprocess_exec
    async def forbidden(*args, **kwargs):
        nonlocal attempted
        attempted = True
        raise AssertionError('competing provider reached process creation; no extra GPU weights are loaded by this check')
    asyncio.create_subprocess_exec = forbidden
    rejected = False
    try:
        try:
            await candidate.open_session(GenerationKey('admission-check', 'never-generated', 0))
        except DeviceBusy:
            rejected = True
        if not rejected or attempted or candidate._process is not None:
            raise ValueError('competing provider was not rejected before spawning')
    finally:
        await candidate.close()
        asyncio.create_subprocess_exec = original
    result = {'standalone_provider_rejected': rejected, 'competing_spawn_attempted': attempted,
        'real_qwen_child_lease_holders': native_holders, 'config_model_revision': config.model_revision,
        'scope': 'cooperating processes using the same operator-owned runtime descriptor directory / device-0 lease'}
    with args.output.open('x') as target:
        target.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('refusing to overwrite evidence')
    asyncio.run(run(args))


if __name__ == '__main__': main()
