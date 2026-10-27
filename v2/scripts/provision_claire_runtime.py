"""Provision the explicitly authorized, isolated TensorRT 10.13.3 CUDA-12 runtime.

Verify the complete published wheel before extracting only Linux libraries.
No SDK/model download scripts, global installation or prebuilt engines are used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WHEEL_URL = 'https://pypi.nvidia.com/tensorrt-cu12-libs/tensorrt_cu12_libs-10.13.3.9-py2.py3-none-manylinux_2_28_x86_64.whl'
WHEEL_SIZE = 3115257094
WHEEL_SHA256 = 'bf2008eca911411aa93b852825ea992de33396451ea11713a6eb97c411d3b2e9'
HEADER_REVISION = '94e2b9ef6d2cce74c76cdad499cca36cc4949197'
PARSER_REVISION = '9a9f7883dd7b8cb0a718395bac2075fab6f97da8'
LIBRARIES = {'libnvinfer.so.10': 673033280, 'libnvinfer_builder_resource.so.10.13.3': 2122364432,
    'libnvinfer_plugin.so.10': 59838160, 'libnvonnxparser.so.10': 4625368}


def parser_header(headers):
    # Exact onnx parser submodule of the pinned TensorRT source revision.
    url=f'https://api.github.com/repos/onnx/onnx-tensorrt/contents/NvOnnxParser.h?ref={PARSER_REVISION}'
    with urllib.request.urlopen(url,timeout=30) as source: entry=json.load(source)
    if entry['type']!='file' or not 0<entry['size']<1024**2: raise ValueError('bounded parser header required')
    url=f'https://raw.githubusercontent.com/onnx/onnx-tensorrt/{PARSER_REVISION}/NvOnnxParser.h'
    with urllib.request.urlopen(url,timeout=30) as source: data=source.read(entry['size']+1)
    if len(data)!=entry['size'] or hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()!=entry['sha']:
        raise ValueError('pinned parser header Git object mismatch')
    with (headers/'NvOnnxParser.h').open('xb') as target: target.write(data)
    return {'path':'include/NvOnnxParser.h','sha256':hashlib.sha256(data).hexdigest(),'size':len(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT / 'var/toolchains/tensorrt-10.13.3.9')
    args = parser.parse_args()
    if os.environ.get('HF_HUB_OFFLINE') == '1' or os.environ.get('TRANSFORMERS_OFFLINE') == '1': raise ValueError('explicit runtime provisioning requires online mode')
    if args.destination.exists(): raise ValueError('preserve existing runtime; verify its recorded hashes instead')
    args.destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if shutil.disk_usage(args.destination.parent).free < WHEEL_SIZE + sum(LIBRARIES.values()) + 4*1024**3:
        raise ValueError('insufficient space for verified wheel, Linux libraries and 4 GiB reserve')
    staging = Path(tempfile.mkdtemp(prefix='.tensorrt-', dir=args.destination.parent))
    started = time.monotonic()
    try:
        wheel = staging / 'runtime.whl'; digest = hashlib.sha256(); count = 0
        with urllib.request.urlopen(WHEEL_URL, timeout=40) as source, wheel.open('xb') as target:
            while block := source.read(min(1024**2, WHEEL_SIZE-count+1)):
                count += len(block)
                if count > WHEEL_SIZE or time.monotonic()-started > 600: raise ValueError('runtime download exceeds byte/time bound')
                digest.update(block); target.write(block)
                if count % (128*1024**2) == 0: print(json.dumps({'downloaded_mib': count//1024**2}), flush=True)
        if count != WHEEL_SIZE or digest.hexdigest() != WHEEL_SHA256: raise ValueError('published runtime wheel size/hash mismatch')
        libraries = staging / 'lib'; libraries.mkdir(mode=0o700)
        with zipfile.ZipFile(wheel) as archive:
            for name, size in LIBRARIES.items():
                info = archive.getinfo('tensorrt_libs/' + name)
                if info.file_size != size or (info.external_attr >> 16) & 0o170000 == 0o120000: raise ValueError('unexpected native runtime member')
                with archive.open(info) as source, (libraries / name).open('xb') as target:
                    shutil.copyfileobj(source, target, 1024**2)
                if (libraries/name).stat().st_size != size: raise ValueError('truncated runtime extraction')
            license_name = 'tensorrt_cu12_libs-10.13.3.9.dist-info/LICENSE.txt'
            license_text = archive.read(license_name)
            if len(license_text) > 128*1024: raise ValueError('bounded runtime license required')
            (staging / 'TensorRT-LICENSE.txt').write_bytes(license_text)
        wheel.unlink()  # Only this verified temporary archive; keep installed Linux libraries and hashes.
        for link, target in [('libnvinfer.so','libnvinfer.so.10'), ('libnvinfer_plugin.so','libnvinfer_plugin.so.10'), ('libnvonnxparser.so','libnvonnxparser.so.10')]:
            (libraries / link).symlink_to(target)
        headers = staging / 'include'; headers.mkdir(mode=0o700)
        url = f'https://api.github.com/repos/NVIDIA/TensorRT/contents/include?ref={HEADER_REVISION}'
        with urllib.request.urlopen(url, timeout=30) as source:
            raw = source.read(1024**2+1)
        if len(raw) > 1024**2: raise ValueError('bounded SDK header inventory required')
        entries = json.loads(raw); records = []
        for entry in entries:
            name = entry['name']
            if entry['type'] != 'file' or not name.endswith('.h'): continue
            if Path(name).name != name or entry['size'] > 1024**2: raise ValueError('unsafe header inventory')
            url = f'https://raw.githubusercontent.com/NVIDIA/TensorRT/{HEADER_REVISION}/include/{name}'
            with urllib.request.urlopen(url, timeout=30) as source: data = source.read(entry['size']+1)
            git_hash = hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
            if len(data) != entry['size'] or git_hash != entry['sha']: raise ValueError('pinned header Git object mismatch')
            (headers / name).write_bytes(data)
            records.append({'path':'include/'+name,'sha256':hashlib.sha256(data).hexdigest(),'size':len(data)})
        if not (headers/'NvInfer.h').exists() or not (headers/'NvInferVersion.h').exists(): raise ValueError('TensorRT public headers missing')
        records.append(parser_header(headers))
        for name in LIBRARIES:
            path=libraries/name; digest=hashlib.sha256()
            with path.open('rb') as source:
                for block in iter(lambda:source.read(1024**2),b''):digest.update(block)
            records.append({'path':'lib/'+name,'sha256':digest.hexdigest(),'size':path.stat().st_size})
        manifest={'schema_version':1,'runtime':'TensorRT CUDA-12 10.13.3.9','wheel_url':WHEEL_URL,
            'wheel_size':WHEEL_SIZE,'wheel_sha256':WHEEL_SHA256,'header_revision':HEADER_REVISION,'parser_revision':PARSER_REVISION,'files':records,
            'excluded':'unused Windows builder resource and Python package files; wheel verified in full before Linux-only extraction',
            'license':'TensorRT-LICENSE.txt retained; binary runtime terms differ from Apache-licensed public headers',
            'model_downloads':False,'engine_downloads':False}
        (staging/'runtime.json').write_text(json.dumps(manifest,indent=2)+'\n')
        staging.rename(args.destination)
        print(json.dumps({'runtime_provisioned':True,'headers':len(records)-len(LIBRARIES),'libraries':len(LIBRARIES),'elapsed_seconds':time.monotonic()-started}))
    finally:
        if staging.exists(): shutil.rmtree(staging)


if __name__ == '__main__': main()
