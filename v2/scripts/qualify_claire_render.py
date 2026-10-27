"""Export full-size recorded/direct GPU parity, ownership and scoped trace evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from talkinghead_v2.local import digest
from talkinghead_v2.session.device import DeviceLease

ROOT=Path(__file__).resolve().parents[1]


def frame_hashes(path):
    output=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-f','framehash','-hash','sha256','-'],text=True,timeout=10)
    return [line.rsplit(',',1)[-1].strip() for line in output.splitlines() if line and not line.startswith('#')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recorded',type=Path,required=True)
    parser.add_argument('--direct',type=Path,required=True)
    parser.add_argument('--no-readback',type=Path,required=True)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--neural-directory',type=Path,required=True)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists():raise ValueError('preserve prior evidence')
    records={}
    for name,directory in [('recorded',args.recorded),('direct',args.direct),('no_readback',args.no_readback)]:
        d=json.loads((directory/'render.json').read_text())
        if any(digest(ROOT/name)!=expected for name,expected in d['source_sha256'].items()):raise ValueError('render source changed after observed run')
        if d['metrics']['encoded_frames']!=84 or d['decoded']['nb_read_frames']!='84' or d['decoded']['has_b_frames']!=0:
            raise ValueError('coded-frame conservation/profile missing')
        if d['metrics']['maximum_tile_faces']>8192 or d['metrics']['maximum_tile_references']>4194304:raise ValueError('bounded compact tile limits exceeded')
        if digest(directory/d['coded_file'])!=d['coded_sha256']:raise ValueError('coded evidence changed')
        records[name]=d
    host=records['recorded'];direct=records['direct'];traced=records['no_readback']
    if host['geometry_host_reference'] is not True or direct['geometry_host_reference'] is not False or traced['qualification_image_readback'] is not False:
        raise ValueError('reference/direct/readback boundaries are mislabeled')
    if direct['binary_sha256']!=traced['binary_sha256'] or direct['ptx_sha256']!=traced['ptx_sha256']:raise ValueError('traced direct runtime differs from parity runtime')
    reference_frames=frame_hashes(args.recorded/host['coded_file'])
    observed_frames=frame_hashes(args.direct/direct['coded_file'])
    if len(reference_frames)!=84 or observed_frames!=reference_frames:raise ValueError('whole decoded video differs between recorded and direct GPU paths')
    snapshots={}
    for index in [0,82]:
        a=args.recorded/f'snapshot-{index}.rgba';b=args.direct/f'snapshot-{index}.rgba'
        if a.read_bytes()!=b.read_bytes():raise ValueError('raw qualification snapshot mismatch')
        snapshots[str(index)]={'rgba_sha256':digest(b),'reference_exact':True}
    scene=json.loads((args.scene/'scene.json').read_text());mesh=args.scene/'research-scene.bin'
    if digest(mesh)!=scene['geometry_sha256'] or direct['scene_sha256']!=scene['geometry_sha256']:raise ValueError('scene identity changed')
    camera=scene['camera'];source=args.neural_directory/'stream_20ms/geometry.bin'
    if digest(source)!=direct['source_neural_sha256']:raise ValueError('neural reference source changed')
    with source.open('rb') as stream:stream.seek(40);world=np.frombuffer(stream.read(67122*3*4),dtype='<f4').reshape(-1,3).astype(np.float64)
    projected=(world@np.asarray(camera['rotation']).reshape(3,3).T)*camera['units_to_meters']+np.asarray(camera['translation_m'])
    projected[:,:2]=(projected[:,:2]/camera['half_extent_m']+1)*511/2
    gpu=np.fromfile(args.direct/'snapshot-projected.bin',dtype='<f8').reshape(-1,3)
    error=np.abs(gpu-projected)
    if gpu.shape!=(67122,3) or float(error[:,:2].max())>1e-8 or float(error[:,2].max())>1e-10:raise ValueError('independent CPU/GPU camera/unit projection failed')
    # Pinned schema/actual trace: include all D2H sizes rather than assuming that
    # SDK metadata transfers cannot exist. Encoded driver/CPU copies remain.
    with sqlite3.connect(f'file:{args.trace.resolve()}?mode=ro',uri=True) as db:
        copies=db.execute('select e.label,m.bytes,count(*) from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on m.copyKind=e.id group by e.label,m.bytes').fetchall()
        kernels=dict(db.execute('select s.value,count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id group by s.value'))
    d2h=[{'bytes':size,'count':count} for kind,size,count in copies if kind=='Device-to-Host']
    if any(item['bytes']>40 for item in d2h) or not any(item['bytes']==20 and item['count']==84 for item in d2h):raise ValueError('no-readback trace contains larger/unexpected device-to-host copies')
    for size in [61520*3*4,5602*3*4]:
        if ('Device-to-Device',size,84) not in copies:raise ValueError('owned SDK geometry copies missing from trace')
    our=['project_large_mesh','bin_large_mesh','scan_large_tiles','fill_large_tiles','render_large_mesh','rgba_to_nv12']
    if any(kernels.get(name)!=84 for name in our) or kernels.get('Convert_PL2BL')!=168:raise ValueError('renderer/conversion frame kernels incomplete')
    contracts={}
    for name in ['tha_large_mesh_contract','tha_encoder_contract']:
        path=ROOT/'var/p03/large-gpu-contracts'/(name+'.json');d=json.loads(path.read_text())
        if d['returncode']:raise ValueError('GPU contract failed')
        contracts[name]=json.loads(d['stdout'])
    suites=list(ET.parse(ROOT/'var/p03/claire-render-core-tests.xml').getroot().iter('testsuite'))
    if sum(int(s.get('tests',0)) for s in suites)!=287 or any(int(s.get(k,0)) for s in suites for k in ['failures','errors','skipped']):raise ValueError('CPU regression incomplete/failed')
    memory=json.loads((args.direct/'memory.json').read_text())
    if memory['query_failures'] or len(memory['samples'])<5:raise ValueError('direct memory sampling incomplete')
    lease=DeviceLease(ROOT/'var/gpu-device-0.lock');lease.acquire('claire-render-post-retirement');lease.release()
    sources=[*sorted((ROOT/'native/media').glob('*')),ROOT/'native/motion/render_probe.cpp',ROOT/'scripts/prepare_claire_render_scene.py',ROOT/'scripts/run_claire_render.py',Path(__file__)]
    report={'schema_version':1,'phase':'P03/P06','phase_status':'in_progress','evidence_category':'local_gpu_observation',
        'scope':'recorded synthetic Qwen -> actual Claire SDK geometry -> full-size source-matched skin/tongue -> CUDA raster/NV12/NVENC; direct GPU vs recorded-host reference',
        'source_sha256':{path.relative_to(ROOT).as_posix():digest(path) for path in sources if path.is_file()},
        'binary_sha256':{'recorded':host['binary_sha256'],'direct':direct['binary_sha256']},'ptx_sha256':direct['ptx_sha256'],
        'geometry':{'vertices':67122,'triangles':134236,'decimated':False,'source_scene_sha256':scene['geometry_sha256'],
            'source_geometry_parity_report_sha256':scene['source_geometry_parity_report_sha256'],'camera':camera,
            'reference_material':'two-sided flat skin/tongue only; no source texture, eyeball/denture meshes or Gaussian appearance'},
        'parity':{'decoded_frames':84,'all_decoded_yuv_frame_sha256_equal':True,'decoded_hash_inventory_sha256':hashlib.sha256(json.dumps(reference_frames).encode()).hexdigest(),
            'snapshots':snapshots,'projection_max_error_pixels':float(error[:,:2].max()),'projection_max_depth_error_m':float(error[:,2].max())},
        'recorded_reference':host['metrics'],'direct_gpu':direct['metrics'],'traced_no_readback':traced['metrics'],
        'trace':{'sqlite_sha256':digest(args.trace),'scope':'complete direct-process run with --no-readback, including model initialization, streaming audio and EOF',
            'our_kernel_counts':{name:kernels[name] for name in our+['Convert_PL2BL']},'all_cuda_d2h':d2h,
            'owned_geometry_d2d':[{'bytes':size,'count':84} for size in [738240,67224]],
            'observed_large_cuda_d2h':False,'zero_copy_claimed':False,
            'copies':'initial model/topology/material/camera H2D, ongoing audio H2D, owned geometry D2D, small SDK/status D2H, encoded CPU/driver copies and internal NVENC conversion remain',
            'instrumentation':'Nsight device-side event completion trace was enabled; possible false dependencies are not used as latency/concurrency evidence'},
        'contracts':contracts,'core_tests_passed':287,
        'memory':{'samples':len(memory['samples']),'query_failures':0,'sampled_total_device_max_mib':max(s['device_used_mib'] for s in memory['samples']),
            'sampled_combined_worker_max_mib':max(s['worker_used_mib'] for s in memory['samples']),
            'scope':'SDK + renderer + NVENC in one process/context; sampled observations, not guaranteed loading peaks or a matched isolated encoder-budget comparison',
            'qwen_coload':False,'capacity_qualified':False},
        'ownership':'one GPU owner thread, borrowed SDK context retained by SDK until encoder destruction; producer events then owned D2D copies; CPU completion protects source lifetime; fixed surface refs',
        'failures_retained':'initial fixed tile-list overflow; compact-list diagnostic still failed until a pageable host-upload/default-stream race was explicitly fenced. No triangles were deleted to pass.',
        'model_impact':'no additional downloads or training; uses the explicitly authorized pinned Claire package/vendor sample and existing Qwen recording',
        'acceptance':{'full_size_neural_render_encode':True,'direct_gpu_reference_parity':True,'scoped_no_large_cuda_readback':True,
            'live_actor_integration':False,'receiver_playback':False,'physical_av':False,'gaussian_retarget':False,'independent_quality':False,'release':False}}
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'neural_rendered_frames':84,'direct_reference_exact':True,'trace_large_d2h_observed':False,'release':False}))


if __name__=='__main__':main()
