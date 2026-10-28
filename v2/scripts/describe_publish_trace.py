"""Pin already-built publisher diagnostics; no source/model/software fetching."""
import argparse,json,os,shutil
from pathlib import Path
from talkinghead_v2.publish_probe import PublisherTraceConfig,PATCH_SHA256,RATE_PATCH_SHA256,MAX_EVENTS
from talkinghead_v2.server.inbound_config import private_json,file_hash,RUST_REVISION

ROOT=Path(__file__).resolve().parents[1]


def main(args):
    if args.output.exists():raise ValueError('preserve existing trace descriptor')
    build=private_json(args.build)
    expected=build.get('patch_sha256')
    patch=ROOT/'patches'/('livekit-publish-rate-control.patch' if expected==RATE_PATCH_SHA256 else 'livekit-publish-stage-trace.patch')
    if (expected not in (PATCH_SHA256,RATE_PATCH_SHA256) or build.get('library_sha256')!=file_hash(args.library) or
        file_hash(patch)!=expected or
        build.get('build_script_sha256')!=file_hash(ROOT/'scripts/build_publish_trace_ffi.py')):
        raise ValueError('current exact source/build/library evidence required')
    args.output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    snapshot=args.output.parent/('publisher-trace-'+file_hash(args.library)[:16]);snapshot.mkdir(mode=0o700)
    lib=snapshot/'lib';lib.mkdir(mode=0o700)
    originals={'executable':args.worker,'ffi_library':args.library,'cpp_library':ROOT/'var/build/livekit-cpp/lib/liblivekit.so','build_report':args.build}
    paths={'executable':snapshot/'tha_claire_session_media','ffi_library':lib/'liblivekit_ffi.so','cpp_library':lib/'liblivekit.so','build_report':snapshot/'build.json'}
    for role,original in originals.items():
        with original.open('rb') as source,open(paths[role],'xb',opener=lambda p,f:os.open(p,f|os.O_NOFOLLOW,0o600 if role=='build_report' else 0o500)) as destination:
            shutil.copyfileobj(source,destination,1024*1024)
    descriptor={'schema_version':1,'purpose':'isolated_publisher_stage_trace','rust_revision':RUST_REVISION,
        'patch_sha256':expected,'abi_version':1,'max_events':MAX_EVENTS,'external_rate_control_opt_in':expected==RATE_PATCH_SHA256,
        'worker_source_sha256':{p:file_hash(ROOT/p) for p in ['native/media/claire_session_worker.cpp','native/media/publish_trace.h','native/media/CMakeLists.txt']},
        'files':{role:{'path':str(p.relative_to(args.output.parent)),'size':p.stat().st_size,'sha256':file_hash(p)} for role,p in paths.items()}}
    with open(args.output,'x',opener=lambda p,f:os.open(p,f|os.O_NOFOLLOW,0o600)) as target:target.write(json.dumps(descriptor,indent=2)+'\n')
    config=PublisherTraceConfig.load(args.output)
    print(json.dumps({'pinned':True,'revision':config.revision,'ffi_sha256':file_hash(config.ffi_library),'serving_runtime':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('library','build','worker','output'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
