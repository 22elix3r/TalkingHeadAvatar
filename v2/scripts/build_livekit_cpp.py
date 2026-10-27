"""Explicit isolated build of the pinned P06 software SDK; never provision models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SDK='df8319db2f459d194ae0bb7a5d04920ce5641f81'
RUST='bae4df2f3d6c8f0f6ce81a7370c445779f1948d2'
SUBMODULES={'yuv-sys/libyuv':'917276084a49be726c90292ff0a6b0a3d571a6af',
            'livekit-protocol/protocol':'9a7c5cf96bbbcbc30a8ce76bca16815ba5b7bb33'}


def output(args,cwd=None): return subprocess.check_output(args,cwd=cwd,text=True).strip()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'var/sources/livekit-cpp-v1.11.0')
    parser.add_argument('--build',type=Path,default=ROOT/'var/build/livekit-cpp')
    parser.add_argument('--rust-sysroot',type=Path,default=Path(os.environ.get('RUSTUP_HOME',str(Path.home()/'.rustup')))/'toolchains/stable-x86_64-unknown-linux-gnu')
    args=parser.parse_args();source=args.source.resolve();build=args.build.resolve()
    if output(['git','rev-parse','HEAD'],source)!=SDK: raise ValueError('unqualified C++ SDK revision')
    if output(['git','rev-parse','HEAD'],source/'client-sdk-rust')!=RUST: raise ValueError('unqualified Rust FFI revision')
    for name,revision in SUBMODULES.items():
        if output(['git','rev-parse','HEAD'],source/'client-sdk-rust'/name)!=revision:
            raise ValueError('required pinned SDK submodule missing: '+name)
    private=ROOT/'var/p06';private.mkdir(parents=True,exist_ok=True);os.chmod(private,0o700)
    build.mkdir(parents=True,exist_ok=True)
    # Resolve already-installed binaries rather than allowing rustup to install or
    # modify the user's global toolchain when it sees the upstream toolchain file.
    sysroot=args.rust_sysroot.resolve()
    compiler={name:str(sysroot/'bin'/name) for name in ['cargo','rustc','rustdoc']}
    for name,path in compiler.items():
        if not Path(path).is_file(): raise ValueError('required existing Rust toolchain is unavailable')
    if not output([compiler['rustc'],'--version']).startswith('rustc 1.99.0 '):
        raise ValueError('this profile requires the already-installed Rust 1.99.0; no automatic install')
    wrapper=private/'cargo-locked.py'
    wrapper.write_text(f'#!{sys.executable}\nimport os,sys\na=sys.argv[1:]\n'
        f'os.execv({compiler["cargo"]!r},[{compiler["cargo"]!r},*a,*( ["--locked"] if a and a[0]=="build" else [] )])\n')
    os.chmod(wrapper,0o700)
    env=dict(os.environ,CARGO_HOME=str(private/'cargo'),RUSTC=compiler['rustc'],
        RUSTDOC=compiler['rustdoc'],CARGO_BUILD_JOBS='4',CARGO_INCREMENTAL='0',CARGO_PROFILE_RELEASE_DEBUG='0',
        CARGO_NET_RETRY='2',TMPDIR=str(private/'tmp'),XDG_CACHE_HOME=str(private/'cache'),
        PATH=str(sysroot/'bin')+os.pathsep+os.environ['PATH'])
    Path(env['TMPDIR']).mkdir(exist_ok=True,mode=0o700)
    lock=source/'client-sdk-rust/Cargo.lock';before=sha(lock)

    def run(command,name,deadline):
        path=private/name
        with path.open('w') as log:
            os.chmod(path,0o600)
            process=subprocess.Popen(command,env=env,stdout=log,stderr=log,start_new_session=True)
            started=time.monotonic()
            try:
                while process.poll() is None:
                    if shutil.disk_usage(build).free<3*1024**3: raise RuntimeError('native build disk reserve reached')
                    if time.monotonic()-started>deadline: raise TimeoutError('bounded SDK build timed out')
                    time.sleep(1)
                if process.returncode: raise RuntimeError(f'SDK build failed; inspect private {name}')
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise
    run(['cmake','-S',str(source),'-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_C_COMPILER=/usr/bin/gcc-14','-DCMAKE_CXX_COMPILER=/usr/bin/g++-14',
        '-DLIVEKIT_VERSION=1.11.0','-DLIVEKIT_BUILD_EXAMPLES=OFF','-DLIVEKIT_BUILD_TESTS=OFF',
        '-DLIVEKIT_LOG_LEVEL=WARN',f'-DCARGO_EXECUTABLE={wrapper}'],'livekit-configure-locked.log',600)
    run(['cmake','--build',str(build),'--parallel','4'],'livekit-build.log',1800)
    if sha(lock)!=before: raise ValueError('SDK Cargo.lock changed; build is not qualified')
    artifacts={p.relative_to(build).as_posix():sha(p) for p in (build/'lib').glob('liblivekit*') if p.is_file() and not p.is_symlink()}
    if not artifacts: raise ValueError('SDK produced no declared library')
    report={'schema_version':1,'sdk_tag':'v1.11.0','sdk_revision':SDK,'rust_revision':RUST,
        'rust_version':output([compiler['rustc'],'--version']),'cargo_lock_sha256':before,
        'artifacts':artifacts,'model_weights_downloaded':False,
        'global_toolchain_changed_by_this_build':False,
        'toolchain_probe_correction':json.loads((private/'toolchain-probe-correction.json').read_text()) if (private/'toolchain-probe-correction.json').exists() else None,
        'nested_submodules':SUBMODULES,
        'cmake_version':output(['cmake','--version']).splitlines()[0],
        'cxx_version':output(['/usr/bin/g++-14','--version']).splitlines()[0]}
    (private/'livekit-build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
