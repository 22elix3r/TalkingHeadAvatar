"""Build a private, offline API software image from the verified host environment.

Selected model/native caches stay external. This image targets the measured
Linux ABI/driver matrix; no base pull, package install or weight fetch occurs.
"""
import argparse,json,os,platform,re,subprocess,sysconfig
from pathlib import Path
from talkinghead_v2.host_image import RootfsStage,parse_ldd
from talkinghead_v2.local import load_profile,safe_read,runtime_fingerprint,private_json
from talkinghead_v2.server.config import Settings
from talkinghead_v2.server.claire_media import ClaireMediaConfig
from talkinghead_v2.providers.tts.native import load_native_config
from talkinghead_v2.server.inbound_config import file_hash

ROOT=Path(__file__).resolve().parents[1]


def main(args):
    if args.directory.exists() or args.report.exists():raise ValueError('preserve prior image build evidence')
    if not re.fullmatch(r'talkinghead-v2-api:[a-z0-9][a-z0-9_.-]{1,63}',args.image):raise ValueError('explicit project-local image tag required')
    existing=subprocess.run(['docker','image','inspect',args.image],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if existing.returncode==0:raise ValueError('preserve existing image tag')
    profile,_=load_profile(ROOT,args.profile);settings=Settings.load(profile/'runtime.json')
    source=safe_read(profile/'runtime-source.json');runtime=ROOT/'runtimes/session-api/.venv'
    if source['mode']!='shared_qualified_v2' or Path(source['source'])!=runtime or source['fingerprint']!=runtime_fingerprint(runtime):
        raise ValueError('already qualified shared API environment required')
    config=ClaireMediaConfig.load(settings.native_media_runtime);native=load_native_config(settings.tts_runtime)
    if not config.trust_external_rate_control:raise ValueError('qualified local Claire publisher policy required')
    args.directory.mkdir(parents=True,mode=0o700)
    for source_file,label in ((Path(__file__),'build-script.py'),(ROOT/'src/talkinghead_v2/host_image.py','stager-source.py')):
        with source_file.open('rb') as input_stream,open(args.directory/label,'xb',opener=lambda p,f:os.open(p,f,0o600)) as saved:
            for block in iter(lambda:input_stream.read(1024*1024),b''):saved.write(block)
    stdlib=Path(sysconfig.get_path('stdlib'))
    allowed=[runtime,stdlib,ROOT/'src',ROOT/'scripts/serve_p04.py',ROOT/'runtimes/session-api/alembic.ini',
        ROOT/'runtimes/session-api/migrations',Path('/usr'),Path('/lib'),Path('/lib64')]
    stage=RootfsStage(args.directory/'rootfs',allowed,[ROOT/'var'])
    for tree in (runtime,stdlib,ROOT/'src',ROOT/'runtimes/session-api/migrations'):stage.copy_tree(tree)
    for path in (Path('/usr/bin/python3.12'),Path('/usr/bin/nvidia-smi'),ROOT/'scripts/serve_p04.py',ROOT/'runtimes/session-api/alembic.ini'):
        stage.copy_file(path)
    # Include the exact dependencies already resolved for the media snapshot.
    # Project-local native/runtime libraries are separate immutable cache mounts.
    dynamic_roots=set()
    for name,path in config.paths.items():
        if name.startswith('host_dependencies/') and not path.is_relative_to(ROOT/'var'):dynamic_roots.add(path)
    driver_roots=[]
    for name in ('libcuda.so.1','libnvidia-ml.so.1','libnvidia-encode.so.1','libnvcuvid.so.1','libnvidia-ptxjitcompiler.so.1'):
        path=Path('/usr/lib64')/name
        if not path.is_file():raise ValueError('qualified host driver component missing')
        driver_roots.append(path)
    # Use the installed driver packages, including their late-loaded libraries.
    # Dependency selection must not depend on a GPU process being active.
    packages=set(subprocess.check_output(['rpm','-qf','--qf','%{NAME}\n',*map(str,driver_roots)],text=True).splitlines())
    driver_packages={}
    for package in sorted(packages):
        if not re.fullmatch('[a-zA-Z0-9_.+-]+',package):raise ValueError('unexpected installed driver package identity')
        driver_packages[package]=subprocess.check_output(['rpm','-q','--qf','%{NAME} %{VERSION}-%{RELEASE}.%{ARCH} %{LICENSE}',package],text=True).strip()
        files=subprocess.check_output(['rpm','-ql',package],text=True).splitlines()
        for value in files:
            path=Path(value)
            if path.is_relative_to('/usr/lib64') and '.so' in path.name and path.is_file():
                dynamic_roots.update((path,Path('/lib64')/path.relative_to('/usr/lib64')))
    for path in dynamic_roots:stage.copy_file(path)
    scan=[]
    for path,row in list(stage.records.items()):
        if row['kind']=='file':
            with Path(path).open('rb') as f:
                if f.read(4)==b'\x7fELF':scan.append(Path(path))
    visited=set();external={};ldd_logs=[]
    while scan:
        path=scan.pop();key=path.resolve()
        if key in visited:continue
        visited.add(key)
        result=subprocess.run(['ldd',str(path)],capture_output=True,text=True,timeout=10)
        text=result.stdout+result.stderr;ldd_logs.append({'path':str(path),'exit':result.returncode,'output':text})
        if result.returncode and 'not a dynamic executable' not in text and 'statically linked' not in text:
            raise ValueError('software dependency inspection failed; private context retained')
        for dependency in parse_ldd(text):
            if dependency.resolve().is_relative_to(ROOT/'var'):
                external[str(dependency)]={'sha256':file_hash(dependency),'size':dependency.stat().st_size}
            else:
                row=stage.copy_file(dependency);scan.append(dependency)
    image_python=str(runtime/'bin/python');entry=str(ROOT/'scripts/serve_p04.py')
    dockerfile='FROM scratch\nCOPY rootfs/ /\nUSER '+str(os.getuid())+':'+str(os.getgid())+'\n'
    dockerfile+='ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n'
    dockerfile+='ENTRYPOINT '+json.dumps([image_python,entry])+'\n'
    (args.directory/'Dockerfile').write_text(dockerfile)
    private_json(args.directory/'software-inventory.json',{'schema_version':1,'files':stage.records,'external_software_dependencies':external,
        'platform':platform.platform(),'python':sysconfig.get_python_version(),'source_runtime_fingerprint':source['fingerprint'],'installed_driver_packages':driver_packages,
        'api_lock_sha256':file_hash(ROOT/'runtimes/session-api/uv.lock'),'selected_model_caches_and_profile_excluded':True,
        'third_party_terms':'Installed package/driver/runtime terms remain separate; this is a private research image, not redistribution clearance.'})
    private_json(args.directory/'dependency-inspection.json',{'rows':ldd_logs})
    with open(args.directory/'docker-build.log','xb',opener=lambda p,f:os.open(p,f,0o600)) as log:
        subprocess.run(['docker','build','--network=none','--pull=false','--tag',args.image,str(args.directory)],stdout=log,stderr=subprocess.STDOUT,check=True)
    image=json.loads(subprocess.check_output(['docker','image','inspect',args.image],text=True))[0]
    # No cache, device, network or credentials are needed for software imports/help.
    smoke=subprocess.run(['docker','run','--rm','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges:true',args.image,'--help'],capture_output=True,text=True,timeout=30)
    with open(args.directory/'image-help.log','x',opener=lambda p,f:os.open(p,f,0o600)) as log:log.write(smoke.stdout+smoke.stderr)
    stable=runtime_fingerprint(runtime)==source['fingerprint']
    report={'schema_version':1,'status':'built' if smoke.returncode==0 and stable else 'failed','image_id':image['Id'],'image_size_bytes':image['Size'],
        'software_file_records':len(stage.records),'software_logical_bytes':sum(r.get('size',0) for r in stage.records.values()),
        'external_software_file_count':len(external),'software_inventory_sha256':file_hash(args.directory/'software-inventory.json'),
        'build_script_sha256':file_hash(Path(__file__)),'stager_sha256':file_hash(ROOT/'src/talkinghead_v2/host_image.py'),
        'help_process_exit':smoke.returncode,'source_runtime_unchanged':stable,'model_downloads':False,'software_downloads':False,
        'real_provider_or_gpu_image_qualified':False,'scope':'Offline software image from qualified host files. Selected model/native caches stay outside; provider startup, GPU access and browser behavior require separate qualification.'}
    private_json(args.report,report);print(json.dumps(report),flush=True)
    return 0 if report['status']=='built' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','report'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--profile',required=True);parser.add_argument('--image',required=True)
    args=parser.parse_args();args.directory=args.directory.resolve();args.report=args.report.resolve()
    raise SystemExit(main(args))
