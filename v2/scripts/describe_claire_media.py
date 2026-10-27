"""Pin the already-qualified local Claire research runtime; never fetch/build."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from talkinghead_v2.assets.artifacts import sha256_file
from talkinghead_v2.server.claire_media import CLAIRE_REVISION,CLAIRE_SDK_REVISION,ClaireMediaConfig
from talkinghead_v2.server.native_media import SDK_REVISION,private_write

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('engine','scene','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('preserve existing pinned runtime')
    qualified=json.loads((ROOT/'benchmarks/reports/p03-claire-live-preroll.json').read_text())
    if not qualified['live_qwen_streaming'] or qualified['cases'][0]['receiver']['missing_source_frame_indices']:
        raise ValueError('complete live source-frame cohort required before runtime pinning')
    for name,expected in qualified['source_sha256'].items():
        if sha256_file(ROOT/name)!=expected:raise ValueError('qualified native source changed: '+name)
    build=ROOT/'var/build/native-media';lk=ROOT/'var/build/livekit-cpp/lib'
    paths={'executable':build/'tha_claire_session_media','ptx':build/'media_kernels.ptx','bridge_library':build/'libtha_claire_bridge.so',
        'livekit_library':lk/'liblivekit.so','ffi_library':lk/'liblivekit_ffi.so','sdk_library':ROOT/'var/build/audio2face-sdk/audio2x-sdk/lib/libaudio2x.so',
        'scene':args.scene/'research-scene.bin','scene_descriptor':args.scene/'scene.json','model_descriptor':args.engine/'model.json',
        'engine_descriptor':args.engine/'engine.json','engine':args.engine/'network.trt','runtime_manifest':ROOT/'var/toolchains/tensorrt-10.13.3.9/runtime.json'}
    for role,field in [('executable','binary_sha256'),('ptx','ptx_sha256'),('bridge_library','bridge_sha256'),('engine','engine_sha256'),('scene','scene_sha256')]:
        if sha256_file(paths[role])!=qualified[field]:raise ValueError('qualified native binary/model/scene changed')
    engine=json.loads(paths['engine_descriptor'].read_text());scene=json.loads(paths['scene_descriptor'].read_text())
    model=json.loads(paths['model_descriptor'].read_text());model_root=Path(model['modelDataPath']).parent
    for name in engine['model_files']:paths['model_files/'+name]=model_root/name
    runtime=json.loads(paths['runtime_manifest'].read_text())
    for entry in runtime['files']:paths['runtime_files/'+entry['path']]=paths['runtime_manifest'].parent/entry['path']
    env=dict(os.environ,LD_LIBRARY_PATH=str(paths['runtime_manifest'].parent/'lib')+':/usr/local/cuda-12.8/lib64')
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
    payload={'schema_version':1,'profile':'claire_vendor_research','sdk_revision':SDK_REVISION,
        'claire_revision':CLAIRE_REVISION,'claire_sdk_revision':CLAIRE_SDK_REVISION,'prebuffer_ms':300,
        'allowed_use':'local_noncommercial_research','consent_evidence':'vendor_assertion','human_enrollment':False,
        'basis_hash':scene['source_geometry_parity_report_sha256'],'topology_hash':scene['topology_material_sha256'],
        'qualification_report_sha256':sha256_file(ROOT/'benchmarks/reports/p03-claire-live-preroll.json'),
        'files':{name:{'path':str(path.resolve(strict=True)),'size':path.stat().st_size,'sha256':sha256_file(path)} for name,path in paths.items()}}
    args.output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    private_write(args.output,(json.dumps(payload,indent=2)+'\n').encode())
    config=ClaireMediaConfig.load(args.output);config.validate_device()
    print(json.dumps({'pinned':True,'files':len(paths),'revision':config.revision,'new_downloads':False,'profile':'claire_vendor_research'}))


if __name__=='__main__':main()
