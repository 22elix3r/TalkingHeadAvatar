"""Reproduce the pinned SDK build with scoped, explicitly provisioned dependencies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parent))
from provision_claire_runtime import WHEEL_SHA256, HEADER_REVISION, PARSER_REVISION
from talkinghead_v2.local import digest

ROOT=Path(__file__).resolve().parents[1]
SDK_REVISION='1ca0f02535ed774f5dbcd724a31cd486368dc783'


def run(command,env,log,timeout):
    with open(log,'ab',opener=lambda path,flags:os.open(path,flags,0o600)) as output:
        subprocess.run(command,env=env,stdout=output,stderr=output,check=True,timeout=timeout)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provision-dependencies',action='store_true',help='explicit software-only network operation; excludes test data/models')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists():raise ValueError('preserve prior build evidence')
    source=ROOT/'var/sources/audio2face-sdk';runtime=ROOT/'var/toolchains/tensorrt-10.13.3.9';cache=ROOT/'var/packman-cache'
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SDK_REVISION:raise ValueError('SDK source revision mismatch')
    subprocess.run(['git','-C',str(source),'diff','--quiet'],check=True)
    d=json.loads((runtime/'runtime.json').read_text())
    if (d['wheel_sha256'],d['header_revision'],d['parser_revision'])!=(WHEEL_SHA256,HEADER_REVISION,PARSER_REVISION):raise ValueError('TensorRT runtime identity mismatch')
    for entry in d['files']:
        if digest(runtime/entry['path'])!=entry['sha256']:raise ValueError('TensorRT runtime content changed')
    tree=ET.parse(source/'deps/target-deps.packman.xml')
    for child in list(tree.getroot()):
        if child.get('name')=='audio2x-sdk-test-data':tree.getroot().remove(child)
    generated=ET.tostring(tree.getroot(),encoding='unicode')
    manifest=source/'deps/tha-target-deps.packman.xml'
    if manifest.exists() and manifest.read_text()!=generated:raise ValueError('preserving unrelated generated dependency manifest')
    if not manifest.exists():manifest.write_text(generated)
    cache.mkdir(parents=True,exist_ok=True,mode=0o700)
    env=dict(os.environ,PM_PACKAGES_ROOT=str(cache),PM_PYTHON_EXT='/usr/bin/python3.12',
        CUDA_CACHE_PATH=str(ROOT/'var/cache/claire-cuda'))
    directory=ROOT/'var/p03';directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    if args.provision_dependencies:
        run([str(source/'tools/packman/packman'),'pull','-t','config=release','--platform','linux-x86_64',str(manifest)],env,directory/'sdk-dependencies.log',180)
    elif not (source/'_deps/target-deps').is_dir():raise ValueError('software dependencies absent; explicit provisioning required')
    expected={'audiofile','benchmark','cnpy','cxxopts','eigen','gtest','magic_enum','nlohmann_json','refl-cpp','tbtsvd','zlib'}
    if {path.name for path in (source/'_deps/target-deps').iterdir()}!=expected:raise ValueError('dependency inventory includes unknown packages')
    build=ROOT/'var/build/audio2face-sdk'
    command=['cmake','-S',str(source),'-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_COMPILER=/usr/bin/g++-14',
        '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc','-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14',
        '-DCUDAToolkit_ROOT=/usr/local/cuda-12.8','-DCMAKE_CUDA_ARCHITECTURES=89',
        '-DCMAKE_CUDA_FLAGS=-I'+str(ROOT/'var/toolchains/cuda-12.8-include'),'-DTENSORRT_ROOT_DIR='+str(runtime)]
    run(command,env,directory/'sdk-configure.log',60)
    run(['cmake','--build',str(build),'--target','audio2x','--parallel','4'],env,directory/'sdk-build.log',300)
    packages={}
    for package in sorted((cache/'chk').glob('*/*')):
        if not package.is_dir():continue
        records=[]
        for path in sorted(package.rglob('*')):
            if path.name=='.packman.lock':continue
            if path.is_symlink():records.append([path.relative_to(package).as_posix(),'symlink',os.readlink(path)])
            elif path.is_file():records.append([path.relative_to(package).as_posix(),digest(path)])
        packages[package.relative_to(cache/'chk').as_posix()]={'files':len(records),'tree_sha256':hashlib.sha256(json.dumps(records,separators=(',',':')).encode()).hexdigest()}
    report={'schema_version':1,'phase':'P03','phase_status':'in_progress','scope':'pinned SDK/isolated TensorRT software build, not full upstream dataset-dependent unit suite',
        'sdk_revision':SDK_REVISION,'source_clean':True,'source_acquisition':'GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout, then detached pinned checkout',
        'excluded':'upstream sample/test-data package, model/emotion download helpers, install-pip-deps target, bundled compiler/Python provisioning',
        'dependency_manifest_sha256':digest(manifest),'original_dependency_manifest_sha256':digest(source/'deps/target-deps.packman.xml'),
        'package_trees':packages,'packman_version':'7.31','packman_python':'existing /usr/bin/python3.12; tested override, not a claim about upstream Python-helper support',
        'runtime':d,'sdk_library_sha256':digest(build/'audio2x-sdk/lib/libaudio2x.so'),'compiler':'g++ 14.3.1 / nvcc 12.8.93, isolated prior CUDA header overlay',
        'cuda_architectures':[89],'abi':'_GLIBCXX_USE_CXX11_ABI=0 as upstream requires',
        'cmake_version_detection':'upstream helpers print TensorRT ? and magic_enum 5.5.5; actual pinned headers/runtime are 10.13.3.9 and package 0.9.5-0',
        'source_sha256':{Path(__file__).relative_to(ROOT).as_posix():digest(Path(__file__))},
        'global_install':False,'model_downloads_by_build':False,'unknown_external_engine_loaded':False,'full_upstream_tests_passed':False,'release':False}
    linked={}
    listing=subprocess.check_output(['ldd',str(build/'audio2x-sdk/lib/libaudio2x.so')],text=True,env=dict(env,LD_LIBRARY_PATH=str(runtime/'lib')+':/usr/local/cuda-12.8/lib64'))
    for line in listing.splitlines():
        if '=> not found' in line:raise ValueError('SDK runtime dependency unresolved')
        if '=>' not in line:continue
        name,right=line.split('=>',1);path=Path(right.strip().split(' ',1)[0])
        if path.is_file():linked[name.strip()]={'sha256':digest(path),'resolved_path':str(path.resolve())}
    report['observed_linked_libraries']=linked
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'sdk_built':True,'packages':len(packages),'model_downloads':False,'release':False}))


if __name__=='__main__':main()
