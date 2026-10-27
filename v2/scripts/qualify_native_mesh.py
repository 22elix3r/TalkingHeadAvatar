"""Run bounded CPU/GPU neutral-raster parity and export scoped trace evidence."""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import xml.etree.ElementTree as ET

from talkinghead_v2.assets.avatar import load_avatar
from talkinghead_v2.contracts import Envelope, GenerationKey, MotionFrame
from talkinghead_v2.local import digest
from talkinghead_v2.rendering.mesh import MeshRenderer
from talkinghead_v2.rendering.projected import ProjectedMesh, project_neutral
from talkinghead_v2.session.device import DeviceLease

ROOT = Path(__file__).resolve().parents[1]


def write_private(path, data):
    with open(path, 'xb', opener=lambda p, flags: os.open(p, flags | os.O_NOFOLLOW, 0o600)) as target:
        target.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.directory.exists() or args.report.exists(): raise ValueError('refusing to overwrite evidence')
    args.directory.mkdir(parents=True, mode=0o700)
    build = ROOT / 'var/build/native-media'
    asset = load_avatar(ROOT / 'src/talkinghead_v2/fixtures/p03-avatar/avatar.json', tenant_id='procedural')
    key = GenerationKey('native-mesh-qualification', 'neutral', 0)
    frame = MotionFrame(Envelope(key, 0, 'qualification'), asset.rig, 0, 48000, 48000, tuple(0. for _ in asset.rig.coefficient_names))
    expected = MeshRenderer(asset, resolution=512).render(frame, duration=Fraction(1, 30)).rgb
    width, height = 16, 32
    vertices = tuple((x * 511 / (width - 1), y * 511 / (height - 1), 1.) for y in range(height) for x in range(width))
    faces = []
    for y in range(height - 1):
        for x in range(width - 1):
            index = y * width + x
            faces.extend(((index, index + 1, index + width), (index + 1, index + width + 1, index + width)))
    dense = ProjectedMesh(512, vertices, tuple(faces))
    cases = {}
    lease = DeviceLease(ROOT / 'var/gpu-device-0.lock'); lease.acquire('native-mesh-qualification')
    try:
        for name, mesh in [('neutral', project_neutral(asset, 512)), ('dense', dense)]:
            directory = args.directory / name; directory.mkdir(mode=0o700)
            geometry, pixels, coded = directory / 'geometry.bin', directory / 'frame.rgba', directory / 'video.h264'
            write_private(geometry, mesh.packed())
            run = subprocess.run([str(build / 'tha_mesh_probe'), str(build / 'media_kernels.ptx'), str(geometry), str(pixels), str(coded)],
                pass_fds=(lease.fileno,), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            write_private(directory / 'process.json', (json.dumps({'returncode': run.returncode, 'stdout': run.stdout, 'stderr': run.stderr}, indent=2) + '\n').encode())
            if run.returncode: raise ValueError('mesh probe failed; inspect private process.json')
            rgba = pixels.read_bytes()
            if len(rgba) != 512 * 512 * 4: raise ValueError('short GPU qualification readback')
            if name == 'neutral':
                rgb = bytes(value for index, value in enumerate(rgba) if index % 4 != 3)
                mismatches = sum(a != b for a, b in zip(rgb, expected, strict=True))
            else:
                mismatches = sum(rgba[(y * 512 + x) * 4:(y * 512 + x + 1) * 4] !=
                    bytes((112, 196, 220, 255) if x < 511 and y < 511 else (18, 24, 34, 255)) for y in range(512) for x in range(512))
            if mismatches or not all(value == 255 for value in rgba[3::4]): raise ValueError('CPU/GPU raster parity failed')
            observed = json.loads(run.stdout)
            if observed['frames'] != 60 or observed['qualification_raw_readback_bytes'] != 1048576:
                raise ValueError('unexpected qualification scope')
            decoded = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-show_entries',
                'stream=codec_name,profile,width,height,pix_fmt,color_range,color_space,nb_read_frames', '-of', 'json', str(coded)], text=True, timeout=10))['streams']
            if len(decoded) != 1 or decoded[0]['nb_read_frames'] != '60' or decoded[0]['width'] != 512:
                raise ValueError('encoded frame decode failed')
            cases[name] = {'probe': observed, 'raster_oracle_mismatches': 0, 'rgba_sha256': digest(pixels),
                'geometry_sha256': digest(geometry), 'coded_sha256': digest(coded), 'decoded_stream': decoded[0],
                'render_gpu_average_ms': observed['render_gpu_total_ms'] / 60}
        contract = subprocess.check_output([str(build / 'tha_encoder_contract'), str(build / 'media_kernels.ptx')],
            pass_fds=(lease.fileno,), stderr=subprocess.PIPE, text=True, timeout=30)
    finally:
        lease.release()
    with sqlite3.connect(f'file:{args.trace.resolve()}?mode=ro', uri=True) as trace:
        kernels = dict(trace.execute('select s.value,count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id group by s.value'))
        copies = list(trace.execute('select e.label,m.bytes,count(*) from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on m.copyKind=e.id group by e.label,m.bytes'))
        first_kernel = trace.execute('select min(start) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
        late_copies = trace.execute('select count(*) from CUPTI_ACTIVITY_KIND_MEMCPY where start>?', (first_kernel,)).fetchone()[0]
        if kernels.get('render_mesh') != 60 or kernels.get('rgba_to_nv12') != 60 or any(kind != 'Host-to-Device' for kind, _, _ in copies) or sum(count for _, _, count in copies) != 3 or late_copies:
            raise ValueError('trace does not establish init-only uploads/no raw-frame readback')
    tests = ET.parse(ROOT / 'var/p06/mesh-core-tests.xml').getroot()
    total = sum(int(suite.get('tests', 0)) for suite in tests.iter('testsuite'))
    if total < 248 or any(int(suite.get(key, 0)) for suite in tests.iter('testsuite') for key in ['failures', 'errors', 'skipped']):
        raise ValueError('CPU projection qualification incomplete/failed')
    sources = [*sorted((ROOT / 'native/media').glob('*')), ROOT / 'src/talkinghead_v2/rendering/projected.py', ROOT / 'tests/test_projected_mesh.py', Path(__file__)]
    report = {'schema_version': 1, 'phase': 'P06', 'phase_status': 'in_progress', 'scope': 'static canonical neutral mesh and first-party synthetic dense raster; no neural motion or real-avatar quality',
        'source_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in sources if path.is_file()},
        'binary_sha256': digest(build / 'tha_mesh_probe'), 'ptx_sha256': digest(build / 'media_kernels.ptx'),
        'canonical_manifest_sha256': asset.manifest_sha256, 'basis_hash': asset.rig.basis_hash, 'topology_hash': asset.rig.topology_hash,
        'cpu_tests': total, 'cpu_junit_sha256': digest(ROOT / 'var/p06/mesh-core-tests.xml'), 'cases': cases, 'gpu_contract': json.loads(contract),
        'trace': {'sha256': digest(args.trace), 'kernels': kernels, 'copies': copies, 'copies_after_first_kernel': late_copies,
            'no_raw_readback_observed': True, 'boundary': 'isolated no-readback dense mesh encoder trace; three init H2D uploads, encoded CPU copies remain'},
        'timing': {'unpaced_isolated_gpu_service': True, 'cold_init_included': False, 'quantiles_qualified': False, 'qwen_coload_qualified': False},
        'hardware_constraint': '128x128 NVENC initialization rejected on this device/profile; explicitly qualified 512x512 output instead',
        'parity_readback': 'one explicit 1 MiB RGBA device-to-host copy per parity case, outside the no-readback trace',
        'acceptance': {'native_product_integration': False, 'real_speaking_avatar': False, 'physical_av_skew': False, 'quality_gate': False, 'release': False},
        'model_impact': 'no weights/reference/model/dependency downloads or changes; generated geometry is a timing fixture'}
    with args.report.open('x') as target: target.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'neutral_parity': True, 'dense_parity': True, 'gpu_service_average_ms': cases['dense']['render_gpu_average_ms'], 'release': False}))


if __name__ == '__main__': main()
