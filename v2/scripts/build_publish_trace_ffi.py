"""Build only the cached, exact SDK publish-stage trace overlay offline."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from build_inbound_audio_ffi import digest, output, verify_tree
from talkinghead_v2.server.inbound_config import RUST_REVISION, CARGO_LOCK_SHA256
from talkinghead_v2.publish_probe import RATE_PATCH_SHA256

ROOT=Path(__file__).resolve().parents[1]
PATCH_SHA256='5aca3d55bcf694000757e675240d4a23df03590953da60aa7fe2638aca0604c6'
NAMES=['livekit/src/room/track/local_video_track.rs','livekit-ffi/src/lib.rs','livekit-ffi/src/tha_publish_trace.rs']


def verify_source(source,variant):
    patch=ROOT/'patches'/('livekit-publish-rate-control.patch' if variant=='rate-control' else 'livekit-publish-stage-trace.patch')
    expected=RATE_PATCH_SHA256 if variant=='rate-control' else PATCH_SHA256
    if digest(patch)!=expected:raise ValueError('exact publisher trace patch required')
    verify_tree(source,RUST_REVISION,patch.read_bytes())
    if digest(source/'Cargo.lock')!=CARGO_LOCK_SHA256:raise ValueError('existing exact Cargo lock required')
    for name,revision in [('livekit-protocol/protocol','9a7c5cf96bbbcbc30a8ce76bca16815ba5b7bb33'),
                          ('yuv-sys/libyuv','917276084a49be726c90292ff0a6b0a3d571a6af')]:
        verify_tree(source/name,revision)
    names=NAMES+(['webrtc-sys/src/passthrough_video_encoder.cpp'] if variant=='rate-control' else [])
    return {name:digest(source/name) for name in names}


def main(args):
    source,target,cargo_home=(p.resolve(strict=True) for p in (args.source,args.target,args.cargo_home))
    if args.report.exists():raise ValueError('preserve existing build evidence')
    before=verify_source(source,args.variant)
    tools={name:args.rust_sysroot.resolve()/'bin'/name for name in ('cargo','rustc','rustdoc')}
    if not all(p.is_file() for p in tools.values()) or not output([str(tools['rustc']),'--version'],source).startswith('rustc 1.99.0 '):
        raise ValueError('existing Rust 1.99.0 required; no automatic install')
    webrtc,protoc=args.webrtc.resolve(strict=True),args.protoc.resolve(strict=True)
    if not (webrtc/'lib/libwebrtc.a').is_file() or not protoc.is_file():raise ValueError('cached WebRTC and protoc required')
    environment=dict(os.environ,CARGO_HOME=str(cargo_home),CARGO_TARGET_DIR=str(target),LK_CUSTOM_WEBRTC=str(webrtc),
        PROTOC=str(protoc),CARGO_NET_OFFLINE='true',CARGO_BUILD_JOBS='4',CARGO_INCREMENTAL='0',
        RUSTC=str(tools['rustc']),RUSTDOC=str(tools['rustdoc']),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    args.report.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    logs=[]
    for name,command in [
        ('tests',['test','--locked','--offline','-p','livekit-ffi','--release','tha_publish_trace::tests','--','--nocapture']),
        ('build',['build','--locked','--offline','-p','livekit-ffi','--release'])]:
        if shutil.disk_usage(target).free<3*1024**3:raise ValueError('build reserve reached')
        log=args.report.parent/(args.report.stem+'-'+name+'.log')
        with open(log,'xb',opener=lambda p,f:os.open(p,f|os.O_NOFOLLOW,0o600)) as destination:
            result=subprocess.run([str(tools['cargo']),*command],cwd=source,env=environment,
                stdin=subprocess.DEVNULL,stdout=destination,stderr=subprocess.STDOUT,timeout=900)
        if result.returncode:raise ValueError('offline trace build failed; inspect private log')
        logs.append(log)
    if '5 passed; 0 failed' not in logs[0].read_text():raise ValueError('five actual trace tests were not selected')
    if before!=verify_source(source,args.variant):raise ValueError('SDK source changed during build')
    report={'schema_version':1,'rust_revision':RUST_REVISION,'cargo_lock_sha256':CARGO_LOCK_SHA256,
        'patch_sha256':RATE_PATCH_SHA256 if args.variant=='rate-control' else PATCH_SHA256,'library_sha256':digest(target/'release/liblivekit_ffi.so'),
        'external_rate_control_opt_in':args.variant=='rate-control',
        'source_sha256':before,'trace_tests_passed':5,'abi_version':1,'max_events':16384,'max_traces':8,
        'build_script_sha256':digest(Path(__file__)),'source_guard_script_sha256':digest(ROOT/'scripts/build_inbound_audio_ffi.py'),
        'rust_version':output([str(tools['rustc']),'--version'],source),'protoc_sha256':digest(protoc),
        'webrtc_sha256':digest(webrtc/'lib/libwebrtc.a'),'logs_sha256':{p.name:digest(p) for p in logs},
        'webrtc_encoder_header_sha256':digest(webrtc/'include/api/video_codecs/video_encoder.h'),
        'model_downloads':False,'offline':True,'scope':'isolated publisher diagnostics; not a serving inbound-runtime replacement'}
    with open(args.report,'x',opener=lambda p,f:os.open(p,f|os.O_NOFOLLOW,0o600)) as target_file:
        target_file.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'built':True,'trace_tests':5,'library_sha256':report['library_sha256']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('trace','rate-control'),default='trace')
    for name in ('source','target','cargo-home','webrtc','protoc','rust-sysroot','report'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
