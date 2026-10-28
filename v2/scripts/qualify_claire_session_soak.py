"""Owned typed-browser soak with bounded read-only host/native resource sampling."""
import argparse,json,os,signal,sys,threading,time
from pathlib import Path
import subprocess
from qualify_claire_receiver_matrix import run_owned_process
from talkinghead_v2.local import load_profile,owned_process,process_identity,private_json
from talkinghead_v2.server.config import Settings
from talkinghead_v2.server.claire_media import ClaireMediaConfig
from talkinghead_v2.providers.tts.native import load_native_config
from talkinghead_v2.resource_probe import parse_gpu_sample,process_sample,native_coload
from talkinghead_v2.server.inbound_config import file_hash

ROOT=Path(__file__).resolve().parents[1]
GPU_FIELDS='index,memory.used,memory.total,utilization.gpu,utilization.encoder,temperature.gpu,clocks.gr,power.draw'


def main(args):
    if args.directory.exists() or args.report.exists():raise ValueError('preserve previous soak evidence')
    profile,_=load_profile(ROOT,args.profile);settings=Settings.load(profile/'runtime.json')
    config=ClaireMediaConfig.load(settings.native_media_runtime);native=load_native_config(settings.tts_runtime)
    api=owned_process(ROOT,profile)
    if not api or not config.trust_external_rate_control:raise ValueError('owned API and qualified publisher policy required')
    children=Path(f'/proc/{api["pid"]}/task/{api["pid"]}/children')
    if children.read_text().split():raise ValueError('profile has active native children; preserve existing conversation')
    sources=['scripts/qualify_claire_session_soak.py','scripts/qualify_claire_receiver_matrix.py','src/talkinghead_v2/resource_probe.py',
        'src/talkinghead_v2/server/actor.py','src/talkinghead_v2/server/claire_media.py','src/talkinghead_v2/server/native_media.py','src/talkinghead_v2/providers/tts/native.py',
        'web/src/client.ts','web/tests/claire-soak.spec.ts','web/playwright.claire.soak.config.ts','web/playwright.claire-profile.ts']
    before={p:file_hash(ROOT/p) for p in sources}
    args.directory.mkdir(parents=True,mode=0o700);stop=threading.Event();samples=[];failures=[];seen={};observed_native={}
    def collect():
        with open(args.directory/'resources.jsonl','x',opener=lambda p,f:os.open(p,f|os.O_NOFOLLOW,0o600)) as out:
            while not stop.is_set():
                begin=time.monotonic_ns()
                try:
                    if process_identity(api['pid'])!=api:raise ValueError('API identity changed')
                    if len(samples)>=20000:raise ValueError('resource sample bound exceeded')
                    gpu=parse_gpu_sample(subprocess.check_output(['nvidia-smi','--query-gpu='+GPU_FIELDS,'--format=csv,noheader,nounits'],text=True,timeout=3))
                    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=3)
                    gpu_by_pid={int(r.split(',')[0]):int(r.split(',')[1]) for r in compute.splitlines() if r.strip()}
                    processes=[{'role':'api','pid':api['pid'],**(process_sample(api['pid'],api['start_ticks']) or {})}]
                    for value in children.read_text().split():
                        pid=int(value);identity=process_identity(pid)
                        if not identity:continue
                        role='media' if identity['argv'][0]==str(config.executable).encode() else 'qwen' if identity['argv'][0]==native.command[0].encode() else None
                        if role is None:continue
                        resource=process_sample(pid,identity['start_ticks'])
                        if resource is None:continue
                        key=(pid,identity['start_ticks']);seen[key]=role
                        processes.append({'role':role,'pid':pid,**resource,'gpu_mib':gpu_by_pid.get(pid)})
                        if role=='media':
                            base=Path('/proc')/value
                            try:
                                loaded={line.split()[-1] for line in (base/'maps').read_text().splitlines() if '/' in line}
                                required={str(config.paths[r]) for r in ('ffi_library','livekit_library','bridge_library')}
                                if required<=loaded:
                                    env=dict(x.split(b'=',1) for x in (base/'environ').read_bytes().split(b'\0') if b'=' in x)
                                    if env.get(b'THA_LK_TRUST_EXTERNAL_RATE_CONTROL')!=b'1' or any(x in env for x in (b'THA_CLAIRE_CAPTURE_TRACE',b'THA_CLAIRE_PUBLISH_TRACE',b'THA_CLAIRE_RATE_PROBE',b'THA_CLAIRE_RATE_PROBE_SCHEDULE')):
                                        raise ValueError('native serving policy mismatch')
                                    observed_native[key]=True
                            except (FileNotFoundError,ProcessLookupError):pass
                    meminfo={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
                    row={'begin_monotonic_ns':begin,'end_monotonic_ns':time.monotonic_ns(),'gpu':gpu,'host_memory_bytes':meminfo,'processes':processes}
                    samples.append(row);out.write(json.dumps(row)+'\n');out.flush()
                except BaseException as error:
                    failures.append({'monotonic_ns':time.monotonic_ns(),'error_type':type(error).__name__})
                    if len(failures)>=100:break
                stop.wait(.2)
    thread=threading.Thread(target=collect);thread.start()
    environment=dict(os.environ,THA_CLAIRE_PROFILE=args.profile,THA_CLAIRE_COHORT=args.cohort,THA_CLAIRE_SOAK_SECONDS=str(args.seconds),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    command=['./node_modules/.bin/playwright','test','-c','playwright.claire.soak.config.ts','--reporter=json']
    os.chdir(ROOT/'web')
    try:
        with open(args.directory/'playwright.json','xb',opener=lambda p,f:os.open(p,f|os.O_NOFOLLOW,0o600)) as log:
            code=run_owned_process(command,environment,log,timeout=args.seconds+240)
    finally:stop.set();thread.join(timeout=10)
    summary_path=ROOT/'var/p06-claire/soak'/args.cohort/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else None
    stable=(before=={p:file_hash(ROOT/p) for p in sources} and ClaireMediaConfig.load(settings.native_media_runtime).revision==config.revision and
        process_identity(api['pid'])==api)
    remaining=children.read_text().split() if children.exists() else ['api_missing']
    co_load=[s for s in samples if native_coload(s['processes'])]
    telemetry={'samples':len(samples),'query_failures':len(failures),'failure_types':sorted({x['error_type'] for x in failures}),
        'sampled_maxima_are_not_guaranteed_peaks':True,'subqueries_are_sequential':True,'target_pause_after_queries_ms':200,
        'co_load_samples':len(co_load),'device_used_sampled_max_mib':max((s['gpu']['memory_used_mib'] for s in samples),default=None),
        'native_process_samples_without_gpu_attribution':sum(p.get('gpu_mib',0) is None for s in samples for p in s['processes']),
        'per_role_sampled_gpu_max_mib':{role:max((p.get('gpu_mib') or 0 for s in samples for p in s['processes'] if p['role']==role),default=0) for role in ('qwen','media')},
        'per_role_sampled_rss_max_bytes':{role:max((p.get('rss_bytes',0) for s in samples for p in s['processes'] if p['role']==role),default=0) for role in ('api','qwen','media')},
        'qwen_process_lifetimes':sum(role=='qwen' for role in seen.values()),'media_process_lifetimes':sum(role=='media' for role in seen.values()),
        'media_lifetimes_with_verified_policy':len(observed_native),'remaining_api_child_count':len(remaining)}
    completed=bool(code==0 and stable and summary and summary.get('passed') and summary.get('close_confirmed') and not remaining and co_load and not failures)
    report={'schema_version':1,'status':'completed' if completed else 'failed','source_sha256':before,'process_exit':code,
        'descriptor_revision':config.revision,'tts_descriptor_sha256':file_hash(settings.tts_runtime),'source_and_runtime_stable':stable,
        'browser_summary':summary,'browser_summary_sha256':file_hash(summary_path) if summary else None,'telemetry':telemetry,
        'raw_resource_sha256':file_hash(args.directory/'resources.jsonl'),'playwright_sha256':file_hash(args.directory/'playwright.json'),
        'co_load_or_desktop_context':args.context,'scope':'Single typed closed-loop actor; sampled host/device/API/native resources. No physical presentation, human speech, open-loop capacity, eight-hour soak or latency SLO qualification.',
        'model_downloads':False,'software_downloads':False,'capacity_qualified':False,'physical_av_qualified':False}
    private_json(args.report,report);print(json.dumps({'status':report['status'],'process_exit':code,'telemetry':telemetry}),flush=True)
    return 0 if completed else 1


if __name__=='__main__':
    def interrupted(signum,frame):raise KeyboardInterrupt('soak interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',required=True);parser.add_argument('--cohort',required=True)
    parser.add_argument('--seconds',type=int,choices=(60,1800),required=True);parser.add_argument('--context',choices=('desktop_only','desktop_and_cpu_packaging'),required=True)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();args.directory=args.directory.resolve();args.report=args.report.resolve()
    raise SystemExit(main(args))
