"""Build a local N=1 Claire engine under device admission; never fetch weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from provision_claire import inputs, ROOT
from provision_claire_runtime import WHEEL_SHA256, HEADER_REVISION, PARSER_REVISION
from talkinghead_v2.assets.pinned import verify
from talkinghead_v2.local import digest
from talkinghead_v2.session.device import DeviceLease


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    if args.directory.exists():raise ValueError('preserve an existing engine/evidence directory')
    revision,entries=inputs('model')
    model=verify(ROOT/'var/artifacts'/('claire-model-'+revision),entries)
    runtime=ROOT/'var/toolchains/tensorrt-10.13.3.9'
    descriptor=json.loads((runtime/'runtime.json').read_text())
    if (descriptor.get('wheel_sha256'),descriptor.get('header_revision'),descriptor.get('parser_revision'))!=(WHEEL_SHA256,HEADER_REVISION,PARSER_REVISION):
        raise ValueError('runtime identity mismatch')
    for entry in descriptor['files']:
        path=runtime/entry['path']
        if path.is_symlink() or path.stat().st_size!=entry['size'] or digest(path)!=entry['sha256']:raise ValueError('runtime hash mismatch')
    binary=ROOT/'var/build/claire-motion/tha_claire_engine'
    sdk=ROOT/'var/sources/audio2face-sdk'
    actual=subprocess.check_output(['git','-C',str(sdk),'rev-parse','HEAD'],text=True).strip()
    if actual!='1ca0f02535ed774f5dbcd724a31cd486368dc783':raise ValueError('SDK revision mismatch')
    subprocess.run(['git','-C',str(sdk),'diff','--quiet'],check=True)
    lease=DeviceLease(ROOT/'var/gpu-device-0.lock');lease.acquire('claire-engine-build')
    try:
        args.directory.mkdir(parents=True,mode=0o700)
        engine=args.directory/'network.trt'
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',
            THA_DEVICE_LEASE_FD=str(lease.fileno),LD_LIBRARY_PATH=str(runtime/'lib')+':/usr/local/cuda-12.8/lib64',
            CUDA_CACHE_PATH=str(ROOT/'var/cache/claire-cuda'))
        Path(env['CUDA_CACHE_PATH']).mkdir(parents=True,exist_ok=True,mode=0o700)
        with open(args.directory/'builder.stderr','xb',opener=lambda path,flags:os.open(path,flags,0o600)) as log:
            run=subprocess.run([str(binary),str(model/'network.onnx'),str(engine.resolve())],pass_fds=(lease.fileno,),
                env=env,stdout=subprocess.PIPE,stderr=log,text=True,timeout=480)
        if run.returncode:raise ValueError('engine build failed; inspect private builder.stderr')
        metrics=json.loads(run.stdout)
        if metrics.get('fixed_batch')!=1 or metrics['engine_bytes']!=engine.stat().st_size:raise ValueError('engine build output invalid')
        properties=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True,timeout=5).strip()
        gpu_uuid,gpu,driver=(part.strip() for part in properties.split(','))
        paths={entry.path:str(model/entry.path) for entry in entries}
        source=json.loads((model/'model.json').read_text())
        source.update(networkInfoPath=paths['network_info.json'],networkPath=str(engine.resolve()),emotionDatabasePath=paths['implicit_emo_db.npz'],
            modelConfigPath=paths['model_config.json'],modelDataPath=paths['model_data.npz'],
            blendshapePaths={'skin':{'config':paths['bs_skin_config.json'],'data':paths['bs_skin.npz']},
                             'tongue':{'config':paths['bs_tongue_config.json'],'data':paths['bs_tongue.npz']}})
        (args.directory/'model.json').write_text(json.dumps(source,indent=2)+'\n');(args.directory/'model.json').chmod(0o600)
        record={'schema_version':1,'model_revision':revision,'model_files':{entry.path:entry.sha256 for entry in entries},
            'runtime_manifest_sha256':digest(runtime/'runtime.json'),'engine_sha256':digest(engine),'model_descriptor_sha256':digest(args.directory/'model.json'),
            'engine_builder_sha256':digest(binary),'sdk_revision':actual,'sdk_library_sha256':digest(ROOT/'var/build/audio2face-sdk/audio2x-sdk/lib/libaudio2x.so'),
            'source_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in [Path(__file__),ROOT/'native/motion/engine_builder.cpp',ROOT/'native/motion/CMakeLists.txt']},
            'gpu':gpu,'gpu_uuid_sha256':hashlib.sha256(gpu_uuid.encode()).hexdigest(),'driver':driver,'cuda_toolkit':'12.8.93',
            'build':metrics,'unknown_external_engine_loaded':False,'model_download_attempted':False,'inference_qualified':False}
        (args.directory/'engine.json').write_text(json.dumps(record,indent=2)+'\n');(args.directory/'engine.json').chmod(0o600)
        print(json.dumps({'engine_built':True,'bytes':metrics['engine_bytes'],'build_ms':metrics['elapsed_ms'],'inference_qualified':False}))
    finally:lease.release()


if __name__=='__main__':main()
