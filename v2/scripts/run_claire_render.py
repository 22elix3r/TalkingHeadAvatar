"""Render exact recorded Claire geometry through the full-size GPU/NVENC profile."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from talkinghead_v2.local import digest
from talkinghead_v2.session.device import DeviceLease

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('recorded','direct'),default='recorded')
    parser.add_argument('--engine',type=Path)
    parser.add_argument('--input-directory',type=Path,default=ROOT/'var/p03/claire-qwen-input')
    parser.add_argument('--no-readback',action='store_true')
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--neural-directory',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    if args.directory.exists():raise ValueError('preserve previous render evidence')
    scene=json.loads((args.scene/'scene.json').read_text());mesh=args.scene/'research-scene.bin'
    if digest(mesh)!=scene['geometry_sha256']:raise ValueError('full-size research scene hash mismatch')
    neural=json.loads((ROOT/'benchmarks/reports/p03-claire-neural.json').read_text())
    geometry=args.neural_directory/'stream_20ms/geometry.bin'
    if digest(geometry)!=neural['cases']['stream_20ms']['geometry_sha256']:raise ValueError('actual neural geometry hash mismatch')
    if scene['source_geometry_parity_report_sha256']!=neural['source_geometry_parity_report_sha256']:raise ValueError('neural/mesh source identity mismatch')
    args.directory.mkdir(parents=True,mode=0o700)
    build=ROOT/'var/build/native-media';ptx=build/'media_kernels.ptx'
    if args.mode=='direct':
        if args.engine is None:raise ValueError('direct mode requires an explicitly built engine directory')
        engine=json.loads((args.engine/'engine.json').read_text())
        if (engine['engine_sha256']!=neural['engine_sha256'] or digest(args.engine/'network.trt')!=engine['engine_sha256'] or
            digest(args.engine/'model.json')!=engine['model_descriptor_sha256'] or
            digest(ROOT/'var/build/audio2face-sdk/audio2x-sdk/lib/libaudio2x.so')!=engine['sdk_library_sha256']):
            raise ValueError('direct SDK/engine provenance changed')
        runtime=ROOT/'var/toolchains/tensorrt-10.13.3.9'
        if digest(runtime/'runtime.json')!=engine['runtime_manifest_sha256']:raise ValueError('direct TensorRT runtime descriptor changed')
        for entry in json.loads((runtime/'runtime.json').read_text())['files']:
            if digest(runtime/entry['path'])!=entry['sha256']:raise ValueError('direct TensorRT content changed')
        input_path=args.input_directory/'audio-16k.f32'
        if digest(input_path)!=neural['recorded_input']['target_pcm_f32_sha256']:raise ValueError('direct source PCM provenance changed')
        binary=build/'tha_claire_render';coded=args.directory/'snapshot.h264'
        command=[str(binary),str(ptx),str(mesh.resolve()),str((args.engine/'model.json').resolve()),str(input_path.resolve()),str((args.directory/'snapshot').resolve())]
        if args.no_readback:command.append('--no-readback')
    else:
        if args.no_readback:raise ValueError('no-readback trace requires the direct GPU path')
        binary=build/'tha_large_mesh_probe';coded=args.directory/'video.h264'
        command=[str(binary),str(ptx),str(mesh.resolve()),str(geometry.resolve()),str(coded.resolve()),str((args.directory/'snapshot').resolve())]
    lease=DeviceLease(ROOT/'var/gpu-device-0.lock');lease.acquire('full-size-claire-render-qualification')
    try:
        env=dict(os.environ,THA_DEVICE_LEASE_FD=str(lease.fileno),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0')
        if args.mode=='direct':env.update(LD_LIBRARY_PATH=str(runtime/'lib')+':/usr/local/cuda-12.8/lib64',CUDA_CACHE_PATH=str(ROOT/'var/cache/claire-cuda'))
        samples=[];query_failures=[];finished=threading.Event()
        def sample(pid):
            while not finished.is_set():
                try:
                    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=3)
                    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=3)
                    worker=0
                    for line in processes.splitlines():
                        process_pid,memory=(int(value.strip()) for value in line.split(','))
                        if process_pid==pid:worker=memory
                    samples.append({'monotonic_ns':time.monotonic_ns(),'device_used_mib':int(gpu.strip()),'worker_used_mib':worker})
                except (ValueError,OSError,subprocess.SubprocessError):query_failures.append(time.monotonic_ns())
                finished.wait(.1)
        with open(args.directory/'stderr.log','xb',opener=lambda path,flags:os.open(path,flags,0o600)) as log:
            process=subprocess.Popen(command,env=env,pass_fds=(lease.fileno,),stdout=subprocess.PIPE,stderr=log,text=True)
            sampling=threading.Thread(target=sample,args=(process.pid,));sampling.start()
            try:stdout,_=process.communicate(timeout=90)
            finally:
                if process.poll() is None:process.kill();process.wait()
                finished.set();sampling.join(timeout=10)
                with open(args.directory/'memory.json','xb',opener=lambda path,flags:os.open(path,flags,0o600)) as target:
                    target.write((json.dumps({'samples':samples,'query_failures':query_failures},indent=2)+'\n').encode())
            run=type('Run',(),{'returncode':process.returncode,'stdout':stdout})()
        with open(args.directory/'process.json','xb',opener=lambda path,flags:os.open(path,flags,0o600)) as target:
            target.write((json.dumps({'returncode':run.returncode,'stdout':run.stdout},indent=2)+'\n').encode())
        if run.returncode:raise ValueError('full-size renderer failed; inspect private process/stderr')
        metrics=json.loads(run.stdout)
        if metrics['encoded_frames']!=84 or (args.mode=='recorded' and (metrics['vertices']!=scene['vertex_count'] or metrics['triangles']!=scene['triangle_count'])):
            raise ValueError('full-size render frame/topology accounting mismatch')
        decoded=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries','stream=codec_name,profile,width,height,pix_fmt,color_range,color_space,nb_read_frames,has_b_frames','-of','json',str(coded)],text=True,timeout=10))['streams']
        if len(decoded)!=1 or decoded[0]['nb_read_frames']!='84' or decoded[0]['width']!=512 or decoded[0]['has_b_frames']!=0:raise ValueError('coded frame decode/profile mismatch')
        for snapshot in sorted(args.directory.glob('snapshot-*.rgba')):
            subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pixel_format','rgba','-video_size','512x512','-i',str(snapshot),'-frames:v','1',str(snapshot.with_suffix('.png'))],check=True,timeout=10)
        sources=[ROOT/'native/media/gpu_encoder.cpp',ROOT/'native/media/gpu_encoder.h',ROOT/'native/media/kernels.cu',ROOT/'native/media/large_probe.cpp',ROOT/'native/media/large_file.h',Path(__file__)]
        if args.mode=='direct':sources.append(ROOT/'native/motion/render_probe.cpp')
        record={'schema_version':1,'source_sha256':{path.relative_to(ROOT).as_posix():digest(path) for path in sources},'binary_sha256':digest(binary),'ptx_sha256':digest(ptx),
            'scene_sha256':digest(mesh),'source_neural_sha256':digest(geometry),'metrics':metrics,'decoded':decoded[0],
            'coded_sha256':digest(coded),'coded_file':coded.name,'snapshots':{path.name:digest(path) for path in args.directory.glob('snapshot-*')},
            'geometry_host_reference':args.mode=='recorded','status_metadata_readback':True,'qualification_image_readback':not args.no_readback,'live_product_integration':False,'physical_av_qualified':False}
        with open(args.directory/'render.json','xb',opener=lambda path,flags:os.open(path,flags,0o600)) as target:target.write((json.dumps(record,indent=2)+'\n').encode())
        print(json.dumps({'rendered_neural_frames':metrics['encoded_frames'],'vertices':scene['vertex_count'],'triangles':scene['triangle_count'],
            'maximum_tile_faces':metrics['maximum_tile_faces'],'maximum_tile_references':metrics['maximum_tile_references'],'render_gpu_average_ms':metrics.get('render_gpu_total_ms',metrics.get('render_gpu_ms'))/metrics['encoded_frames']}))
    finally:lease.release()


if __name__=='__main__':main()
