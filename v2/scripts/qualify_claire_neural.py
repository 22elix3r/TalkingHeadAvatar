"""Actual pinned Claire GPU partition/EOF/cancel tests on recorded synthetic Qwen."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from provision_claire import inputs, ROOT
from talkinghead_v2.assets.pinned import verify
from talkinghead_v2.local import digest
from talkinghead_v2.session.device import DeviceLease


def read_geometry(path):
    with path.open('rb') as stream:
        if stream.read(8)!=b'THACLR1\0':raise ValueError('neural dump version mismatch')
        counts=struct.unpack('<4I',stream.read(16))
    if counts!=(184560,16806,16,6):raise ValueError('neural geometry/jaw/eyes dimensions mismatch')
    dtype=np.dtype([('sample','<i8'),('next_sample','<i8'),('values','<f4',(sum(counts),))])
    if (path.stat().st_size-24)%dtype.itemsize:raise ValueError('truncated neural frame')
    data=np.memmap(path,mode='r',dtype=dtype,offset=24)
    if len(data)>1800 or not np.isfinite(data['values']).all():raise ValueError('unbounded/nonfinite neural output')
    expected=np.arange(len(data),dtype=np.int64)*16000//60
    if not np.array_equal(data['sample'],expected) or not np.array_equal(data['next_sample'],(np.arange(len(data),dtype=np.int64)+1)*16000//60):
        raise ValueError('neural frame sample/timebase mapping mismatch')
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.directory.exists() or args.report.exists():raise ValueError('preserve prior evidence')
    engine=json.loads((args.engine/'engine.json').read_text())
    for path,expected in [(args.engine/'network.trt',engine['engine_sha256']),(args.engine/'model.json',engine['model_descriptor_sha256']),
        (ROOT/'var/build/claire-motion/tha_claire_engine',engine['engine_builder_sha256']),
        (ROOT/'var/build/audio2face-sdk/audio2x-sdk/lib/libaudio2x.so',engine['sdk_library_sha256'])]:
        if digest(path)!=expected:raise ValueError('engine/runtime/builder provenance hash mismatch')
    for path,expected in engine['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('engine builder source changed; build a fresh pinned engine')
    model_revision,entries=inputs('model');verify(ROOT/'var/artifacts'/('claire-model-'+model_revision),entries)
    if engine['model_files']!={entry.path:entry.sha256 for entry in entries}:raise ValueError('engine source model inventory mismatch')
    asset_report_path=ROOT/'benchmarks/reports/p03-claire-assets.json'
    asset_report=json.loads(asset_report_path.read_text())
    if not asset_report['acceptance']['vertex_identity_parity'] or any(digest(ROOT/name)!=expected for name,expected in asset_report['source_sha256'].items()):
        raise ValueError('source geometry parity evidence missing or stale')
    runtime=ROOT/'var/toolchains/tensorrt-10.13.3.9'
    if digest(runtime/'runtime.json')!=engine['runtime_manifest_sha256']:raise ValueError('runtime changed since engine build')
    for entry in json.loads((runtime/'runtime.json').read_text())['files']:
        if digest(runtime/entry['path'])!=entry['sha256']:raise ValueError('runtime file changed')
    uuid,gpu,driver=(part.strip() for part in subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True,timeout=5).strip().split(','))
    if (hashlib.sha256(uuid.encode()).hexdigest(),driver)!=(engine['gpu_uuid_sha256'],engine['driver']):raise ValueError('engine GPU/driver compatibility changed')
    input_dir=ROOT/'var/p03/claire-qwen-input';source=input_dir/'audio-16k.f32';provenance=json.loads((input_dir/'input.json').read_text())
    old=json.loads((ROOT/'var/qualification/p03-qwen-short/report.json').read_text())
    wave=ROOT/'var/qualification/native-waves-serial/native-000.wav'
    if (digest(wave)!=old['source']['source_wave_sha256'] or digest(wave)!=provenance['source_wave_sha256'] or
        digest(source)!=provenance['target_pcm_f32_sha256'] or provenance['target_samples']!=44430 or source.stat().st_size!=44430*4):
        raise ValueError('recorded synthetic Qwen source provenance mismatch')
    args.directory.mkdir(parents=True,mode=0o700)
    lease=DeviceLease(ROOT/'var/gpu-device-0.lock');lease.acquire('claire-neural-qualification')
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',THA_DEVICE_LEASE_FD=str(lease.fileno),CUDA_VISIBLE_DEVICES='0',
        LD_LIBRARY_PATH=str(runtime/'lib')+':/usr/local/cuda-12.8/lib64',CUDA_CACHE_PATH=str(ROOT/'var/cache/claire-cuda'))
    stop=threading.Event();memory=[];memory_errors=[]
    def sample():
        while not stop.is_set():
            try:
                row=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=3)
                used,total=(int(value.strip()) for value in row.strip().split(','))
                memory.append({'monotonic_ns':time.monotonic_ns(),'used_mib':used,'total_mib':total})
            except (ValueError,OSError,subprocess.SubprocessError):memory_errors.append(time.monotonic_ns())
            stop.wait(.1)
    sampling=threading.Thread(target=sample);sampling.start()
    cases={};frames={}
    try:
        for name,chunk,cancel in [('whole_input_reference',480000,0),('stream_20ms',320,0),('irregular_grid_32ms',512,0),
                                  ('cancel_at_one_second',320,16000),('fresh_after_cancel',320,0)]:
            directory=args.directory/name;directory.mkdir(mode=0o700)
            geometry=directory/'geometry.bin'
            with open(directory/'stderr.log','xb',opener=lambda p,f:os.open(p,f,0o600)) as log:
                run=subprocess.run([str(ROOT/'var/build/claire-motion/tha_claire_geometry'),str((args.engine/'model.json').resolve()),str(source),
                    str(geometry.resolve()),str(chunk),str(cancel)],env=env,pass_fds=(lease.fileno,),stdout=subprocess.PIPE,stderr=log,text=True,timeout=60)
            (directory/'process.json').write_text(json.dumps({'returncode':run.returncode,'stdout':run.stdout},indent=2)+'\n');(directory/'process.json').chmod(0o600)
            if run.returncode:raise ValueError('neural case failed; inspect private process/stderr')
            result=json.loads(run.stdout);data=read_geometry(geometry)
            if len(data)!=result['frames'] or result['cancelled']!=bool(cancel):raise ValueError('neural frame/cancel accounting failed')
            if cancel:
                if result['offered_samples']!=16000 or len(data)>=167:raise ValueError('cancel did not bound neural output')
            else:
                if len(data)!=167 or int(data['sample'][-1])>=44430 or result['frames_before_eof']!=152:
                    raise ValueError('neural EOF target/frame accounting failed')
                if chunk<=512 and result['peak_retained_audio_samples']>16000+8320+chunk:
                    raise ValueError('streaming retained audio exceeds explicit SDK pool/context/quantum bound')
            cases[name]={'probe':result,'geometry_sha256':digest(geometry),'gpu_process_retired':True}
            frames[name]=data
    finally:
        stop.set();sampling.join(timeout=5);lease.release()
        (args.directory/'memory.json').write_text(json.dumps({'samples':memory,'query_failures':memory_errors},indent=2)+'\n');(args.directory/'memory.json').chmod(0o600)
    reference=frames['whole_input_reference'];parity={}
    for name in ['stream_20ms','irregular_grid_32ms','fresh_after_cancel']:
        data=frames[name]
        difference=np.asarray(data['values'],dtype=np.float64)-np.asarray(reference['values'],dtype=np.float64)
        maximum=float(np.max(np.abs(difference)));rms=float(np.sqrt(np.mean(difference*difference)))
        if maximum>1e-5:raise ValueError('neural reference/streamed geometry parity failed')
        parity[name]={'maximum_absolute_difference_in_native_units':maximum,'rms_in_native_units':rms,
            'same_frame_sample_ranges':True,'tolerance':1e-5}
    changes=float(np.max(np.abs(reference['values'][-1,:184560]-reference['values'][0,:184560])))
    if changes<.01:raise ValueError('neural skin output did not vary with the speech input')
    if not memory or memory_errors:raise ValueError('neural memory sampling incomplete')
    lease.acquire('claire-neural-post-retirement');lease.release()
    sources=[Path(__file__),ROOT/'native/motion/geometry_probe.cpp',ROOT/'native/motion/engine_builder.cpp',ROOT/'native/motion/CMakeLists.txt']
    report={'schema_version':1,'phase':'P03','phase_status':'in_progress','evidence_category':'local_gpu_observation',
        'scope':'actual pinned Claire regression SDK with recorded synthetic Qwen; whole-input vs incremental execution/EOF/cancel; explicit full geometry readback',
        'source_sha256':{path.relative_to(ROOT).as_posix():digest(path) for path in sources},
        'model_revision':model_revision,'engine_sha256':engine['engine_sha256'],'sdk_revision':engine['sdk_revision'],
        'source_geometry_parity_report_sha256':digest(asset_report_path),
        'sdk_library_sha256':engine['sdk_library_sha256'],'geometry_probe_sha256':digest(ROOT/'var/build/claire-motion/tha_claire_geometry'),
        'runtime_manifest_sha256':engine['runtime_manifest_sha256'],'engine_build':engine['build'],'gpu':gpu,'driver':driver,
        'recorded_input':provenance,'geometry_dimensions':{'skin_vertices':61520,'tongue_vertices':5602,'jaw_transform_floats':16,'eyes_rotation_floats':6},
        'cases':cases,'parity':parity,'skin_first_last_max_absolute_change_cm':changes,
        'clock':{'input_rate':16000,'target_fps_numerator':60,'target_fps_denominator':1,'frame_pts':'floor(frame_index*16000/60) integer source samples',
                 'total_context_samples':8320,'future_context_samples':4160,'future_context_ms':260,'eof_model_context_is_playable_audio':False},
        'memory':{'samples':len(memory),'query_failures':0,'sampled_total_device_used_max_mib':max(row['used_mib'] for row in memory),
                  'scope':'sequential device NVML samples include other processes; sampled maximum is not guaranteed peak',
                  'qwen_coload':False,'render_encode_coload':False},
        'cancellation':'executor/accumulator reset at 16000 offered samples; process retired; fresh process reproduces reference. Live priority interruption and receiver suppression remain unqualified.',
        'measurement':'unpaced offline source partition check; elapsed includes init, numeric-data load, explicit geometry readback and private file writes; no streaming latency/FPS/SLO claim',
        'model_impact':'specific user-authorized Claire package/runtime and vendor sample provisioned; no additional TTS/ASR/LLM/emotion model or unknown engine downloaded',
        'acceptance':{'actual_neural_motion':True,'numeric_partition_parity':True,'source_vertex_identity':True,'live_motion_provider':False,
                      'rendered_quality':False,'gaussian_retarget':False,'physical_av':False,'capacity':False,'quality_gate':False,'release':False}}
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'neural_cases_passed':len(cases),'frames_per_complete_case':167,'maximum_parity_error':max(value['maximum_absolute_difference_in_native_units'] for value in parity.values()),'release':False}))


if __name__=='__main__':main()
