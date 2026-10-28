"""Pin the already-qualified local Claire research runtime; never fetch/build."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from talkinghead_v2.assets.artifacts import sha256_file
from talkinghead_v2.server.claire_media import CLAIRE_PROTOCOL,CLAIRE_REVISION,CLAIRE_SDK_REVISION,ClaireMediaConfig
from talkinghead_v2.server.native_media import SDK_REVISION,private_write
from talkinghead_v2.publish_probe import PublisherTraceConfig
from talkinghead_v2.server.claire_publisher import PUBLISHER_ROLES

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('engine','scene','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--allow-receiver-video-gaps',action='store_true',help='experimental research only: retain observed decoder gaps without claiming quality/release acceptance')
    parser.add_argument('--qualified-live-report',type=Path,default=ROOT/'benchmarks/reports/p03-claire-live-preroll.json')
    parser.add_argument('--publisher-runtime',type=Path,help='explicit qualified SDK/worker probe inventory to snapshot into serving')
    parser.add_argument('--trust-encoded-rate-control',action='store_true',help='explicit local research policy; requires lifecycle/live/rate and clean owner evidence')
    for name in ('lifecycle-report','rate-report','live-owner','lifecycle-owner'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('preserve existing pinned runtime')
    publisher=PublisherTraceConfig.load(args.publisher_runtime) if args.publisher_runtime else None
    supplied=[args.lifecycle_report,args.rate_report,args.live_owner,args.lifecycle_owner]
    if publisher is not None:
        if not args.trust_encoded_rate_control or not all(supplied) or not publisher.supports_external_rate_control:
            raise ValueError('explicit tested publisher policy and complete evidence paths required')
    elif args.trust_encoded_rate_control or any(supplied):raise ValueError('publisher evidence requires an explicit publisher runtime')
    qualified=json.loads(args.qualified_live_report.read_text())
    observed=qualified['cases'][0];received=observed['receiver']['video'];missing=observed['receiver']['missing_source_frame_indices']
    if (not qualified['live_qwen_streaming'] or qualified.get('protocol')!=CLAIRE_PROTOCOL or
        qualified.get('eligible_motion_start_buffer_ms')!=500 or qualified.get('motion_input_lead_bound_ms')!=800 or
        not received or received[0]['source_sample']!=0 or received[-1]['source_sample']!=(observed['sender']['valid_samples']-1)//1600*1600):
        raise ValueError('qualified live source edges and motion buffer policy required before pinning')
    if missing and not args.allow_receiver_video_gaps:
        raise ValueError('decoder gaps remain; experimental research pinning requires --allow-receiver-video-gaps')
    for name,expected in qualified['source_sha256'].items():
        if sha256_file(ROOT/name)!=expected:raise ValueError('qualified native source changed: '+name)
    build=ROOT/'var/build/native-media';lk=ROOT/'var/build/livekit-cpp/lib'
    paths={'executable':build/'tha_claire_session_media','ptx':build/'media_kernels.ptx','bridge_library':build/'libtha_claire_bridge.so',
        'livekit_library':lk/'liblivekit.so','ffi_library':lk/'liblivekit_ffi.so','sdk_library':ROOT/'var/build/audio2face-sdk/audio2x-sdk/lib/libaudio2x.so',
        'scene':args.scene/'research-scene.bin','scene_descriptor':args.scene/'scene.json','model_descriptor':args.engine/'model.json',
        'engine_descriptor':args.engine/'engine.json','engine':args.engine/'network.trt','runtime_manifest':ROOT/'var/toolchains/tensorrt-10.13.3.9/runtime.json'}
    if publisher:
        paths.update(executable=publisher.executable,ffi_library=publisher.ffi_library,livekit_library=publisher.cpp_library,
            publisher_build_report=args.publisher_runtime.parent/publisher.files['build_report']['path'],
            publisher_live_report=args.qualified_live_report,publisher_lifecycle_report=args.lifecycle_report,
            publisher_rate_report=args.rate_report,publisher_live_owner=args.live_owner,publisher_lifecycle_owner=args.lifecycle_owner)
    for role,field in [('executable','binary_sha256'),('ptx','ptx_sha256'),('bridge_library','bridge_sha256'),('engine','engine_sha256'),('scene','scene_sha256')]:
        if sha256_file(paths[role])!=qualified[field]:raise ValueError('qualified native binary/model/scene changed')
    args.output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    snapshot=args.output.parent/('native-'+sha256_file(paths['executable'])[:16]+'-'+sha256_file(paths['bridge_library'])[:16]+'-'+sha256_file(paths['ffi_library'])[:16])
    roles=('executable','ptx','bridge_library','livekit_library','ffi_library')+(tuple(sorted(PUBLISHER_ROLES)) if publisher else ())
    if not snapshot.exists():snapshot.mkdir(mode=0o700)
    for role in roles:
        original=paths[role];target=snapshot/(role+'.json' if role in PUBLISHER_ROLES else original.name)
        if not target.exists():
            with original.open('rb') as source,open(target,'xb',opener=lambda name,flags:os.open(name,flags,0o700 if role=='executable' else 0o600)) as destination:
                shutil.copyfileobj(source,destination,1024*1024)
        if target.is_symlink() or sha256_file(target)!=sha256_file(original):raise ValueError('preserve changed/incomplete native snapshot')
        paths[role]=target
    if {path.name for path in snapshot.iterdir()}!={paths[role].name for role in roles}:raise ValueError('native snapshot inventory changed')
    engine=json.loads(paths['engine_descriptor'].read_text());scene=json.loads(paths['scene_descriptor'].read_text())
    model=json.loads(paths['model_descriptor'].read_text());model_root=Path(model['modelDataPath']).parent
    for name in engine['model_files']:paths['model_files/'+name]=model_root/name
    runtime=json.loads(paths['runtime_manifest'].read_text())
    for entry in runtime['files']:paths['runtime_files/'+entry['path']]=paths['runtime_manifest'].parent/entry['path']
    env=dict(os.environ,LD_LIBRARY_PATH=str(snapshot.resolve())+':'+str(paths['runtime_manifest'].parent/'lib')+':/usr/local/cuda-12.8/lib64')
    env.pop('LD_PRELOAD',None);env.pop('LD_AUDIT',None)
    # Known first-party/qualified binaries only. Pin the resolved host support
    # libraries too; this does not install/change the host's CUDA or driver.
    for role in ('executable','bridge_library','livekit_library','ffi_library','sdk_library'):
        output=subprocess.check_output(['ldd',str(paths[role].resolve())],env=env,text=True,timeout=10)
        if 'not found' in output:raise ValueError('qualified dynamic dependency unavailable')
        for line in output.splitlines():
            if '=>' not in line:continue
            name,location=line.split('=>',1);location=location.strip().split(' (',1)[0]
            if location.startswith('/'):
                key='host_dependencies/'+name.strip();resolved=Path(location).resolve(strict=True)
                if key in paths and paths[key].resolve()!=resolved:raise ValueError('dynamic dependency resolves inconsistently')
                paths[key]=resolved
    payload={'schema_version':1,'profile':'claire_vendor_research','protocol':CLAIRE_PROTOCOL,'sdk_revision':SDK_REVISION,
        'claire_revision':CLAIRE_REVISION,'claire_sdk_revision':CLAIRE_SDK_REVISION,'prebuffer_ms':500,
        'allowed_use':'local_noncommercial_research','consent_evidence':'vendor_assertion','human_enrollment':False,
        'experimental_receiver_video_gaps':missing,'visual_quality_qualified':False,'physical_av_qualified':False,'release_qualified':False,
        'basis_hash':scene['source_geometry_parity_report_sha256'],'topology_hash':scene['topology_material_sha256'],
        'qualification_report_sha256':sha256_file(args.qualified_live_report),
        'publisher_rate_control':'external_trusted_local_research' if publisher else 'sdk_default',
        'files':{name:{'path':str(path.resolve(strict=True)),'size':path.stat().st_size,'sha256':sha256_file(path)} for name,path in paths.items()}}
    args.output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    private_write(args.output,(json.dumps(payload,indent=2)+'\n').encode())
    config=ClaireMediaConfig.load(args.output);config.validate_device()
    print(json.dumps({'pinned':True,'files':len(paths),'revision':config.revision,'new_downloads':False,'profile':'claire_vendor_research'}))


if __name__=='__main__':main()
