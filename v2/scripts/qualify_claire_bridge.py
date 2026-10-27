"""Explicit cached-GPU qualification of the bounded C ABI motion bridge."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from talkinghead_v2.local import digest
from talkinghead_v2.session.device import DeviceLease

ROOT=Path(__file__).resolve().parents[1]


def decoded_hashes(path):
    output=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-f','framemd5','-'],text=True,timeout=30)
    return [line.rsplit(',',1)[1].strip() for line in output.splitlines() if line and not line.startswith('#')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--input-directory',type=Path,required=True)
    parser.add_argument('--reference-directory',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.directory.exists():raise ValueError('preserve previous bridge evidence')
    neural=json.loads((ROOT/'benchmarks/reports/p03-claire-neural.json').read_text())
    engine=json.loads((args.engine/'engine.json').read_text());scene=json.loads((args.scene/'scene.json').read_text())
    runtime=ROOT/'var/toolchains/tensorrt-10.13.3.9';sdk=ROOT/'var/build/audio2face-sdk/audio2x-sdk/lib/libaudio2x.so'
    if (engine['engine_sha256']!=neural['engine_sha256'] or digest(args.engine/'network.trt')!=engine['engine_sha256'] or
        digest(args.engine/'model.json')!=engine['model_descriptor_sha256'] or digest(sdk)!=engine['sdk_library_sha256'] or
        digest(runtime/'runtime.json')!=engine['runtime_manifest_sha256']):raise ValueError('Claire engine/SDK/runtime provenance changed')
    for entry in json.loads((runtime/'runtime.json').read_text())['files']:
        if digest(runtime/entry['path'])!=entry['sha256']:raise ValueError('TensorRT runtime content changed')
    mesh=args.scene/'research-scene.bin';pcm=args.input_directory/'audio-16k.f32'
    if (digest(mesh)!=scene['geometry_sha256'] or digest(pcm)!=neural['recorded_input']['target_pcm_f32_sha256'] or
        scene['source_geometry_parity_report_sha256']!=neural['source_geometry_parity_report_sha256']):raise ValueError('scene/input/geometry identity changed')
    reference=json.loads((args.reference_directory/'render.json').read_text());reference_video=args.reference_directory/reference['coded_file']
    if digest(reference_video)!=reference['coded_sha256'] or reference['scene_sha256']!=digest(mesh):raise ValueError('reference render provenance changed')
    expected=decoded_hashes(reference_video)
    if len(expected)!=84:raise ValueError('complete actual neural reference required')
    args.directory.mkdir(parents=True,mode=0o700)
    build=ROOT/'var/build/native-media';binary=build/'tha_claire_bridge_probe';library=build/'libtha_claire_bridge.so';ptx=build/'media_kernels.ptx'
    exported=subprocess.check_output(['nm','-D','--defined-only',str(library)],text=True)
    symbols=[line.split()[-1] for line in exported.splitlines() if line.strip()]
    if sorted(symbols)!=sorted(['tha_claire_'+name for name in ('open','push','render','get_metrics','rate','hold','preroll','close')]):raise ValueError('only the eight C ABI exports are permitted')
    cases={}
    lease=DeviceLease(ROOT/'var/gpu-device-0.lock');lease.acquire('claire-c-abi-clock-qualification')
    try:
        env=dict(os.environ,THA_DEVICE_LEASE_FD=str(lease.fileno),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',
            LD_LIBRARY_PATH=str(runtime/'lib')+':/usr/local/cuda-12.8/lib64',CUDA_CACHE_PATH=str(ROOT/'var/cache/claire-cuda'))
        for name in ('normal','paced','irregular','short','late','cancel','overflow','invalid','ordering','idle','preroll'):
            directory=args.directory/name;directory.mkdir(mode=0o700);coded=directory/'video.h264'
            command=[str(binary),str(ptx),str(mesh.resolve()),str((args.engine/'model.json').resolve()),str(pcm.resolve()),str(coded.resolve()),name]
            with open(directory/'stderr.log','xb',opener=lambda path,flags:os.open(path,flags,0o600)) as log:
                run=subprocess.run(command,env=env,pass_fds=(lease.fileno,),stdout=subprocess.PIPE,stderr=log,text=True,timeout=60)
            (directory/'process.json').write_text(json.dumps({'returncode':run.returncode,'stdout':run.stdout},indent=2)+'\n')
            if run.returncode:raise ValueError('Claire bridge failed in '+name+'; inspect private diagnostics')
            case=json.loads(run.stdout);cases[name]={'metrics':case}
            if coded.exists():
                hashes=decoded_hashes(coded);cases[name].update(coded_sha256=digest(coded),decoded_frames=len(hashes))
                if len(hashes)!=(74 if name=='late' else 1 if name=='short' else 87 if name in ('idle','preroll') else 84):raise ValueError('bridge decoded-frame conservation failed')
                if name not in ('late','short','preroll'):
                    if hashes[:84]!=expected:raise ValueError('bridge differs from actual direct GPU reference in '+name)
                    cases[name]['all_source_decoded_frames_equal_direct_reference']=True
    finally:lease.release()
    # A retired GPU process may not leave an admission lock behind.
    lease.acquire('claire-bridge-post-exit-check');lease.release()
    sources=['native/motion/claire_bridge.cpp','native/motion/claire_bridge.h','native/motion/claire_bridge.map',
        'native/motion/claire_bridge_probe.cpp','native/media/gpu_encoder.cpp','native/media/gpu_encoder.h','native/media/kernels.cu','native/media/CMakeLists.txt','scripts/qualify_claire_bridge.py']
    report={'schema_version':1,'observation':'local cached-model C ABI / bounded GPU ownership qualification',
        'source_sha256':{path:digest(ROOT/path) for path in sources},'binary_sha256':digest(binary),'library_sha256':digest(library),'ptx_sha256':digest(ptx),
        'engine_sha256':engine['engine_sha256'],'sdk_library_sha256':digest(sdk),'scene_sha256':digest(mesh),'source_pcm_sha256':digest(pcm),
        'caller_cxx_abi':1,'sdk_cxx_abi':0,'exported_symbols':symbols,'queue_bound_frames':32,'queue_bound_duration_ms':32*1000/30,
        'owned_pose_storage_bytes':32*67122*3*4,'immutable_neutral_init_host_upload_bytes':67122*3*4,
        'render_grid_samples_48k':1600,'model_cadence_hz':60,'render_cadence_hz':30,
        'neural_to_render_max_rounding_samples_48k':2,'cases':cases,'admission_reacquired_after_exit':True,
        'additional_downloads':False,'livekit_session_integration':False,'physical_av_qualified':False,'qwen_coload_qualified':False}
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':list(cases),'decoded_reference_equal_cases':['normal','paced','irregular','cancel'],'queue_bound_frames':32,'report':str(args.report)}))


if __name__=='__main__':main()
