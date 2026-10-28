"""Own probes-disabled lifecycle/live checks and bind exit receipts to report bytes."""
import argparse,json,os,signal,sys
from pathlib import Path
from qualify_claire_receiver_matrix import run_owned_process
from talkinghead_v2.publish_probe import PublisherTraceConfig
from talkinghead_v2.server.inbound_config import InboundAudioConfig,file_hash

ROOT=Path(__file__).resolve().parents[1]


def main(args):
    if args.directory.exists() or args.report.exists():raise ValueError('preserve prior publisher lifecycle evidence')
    publisher=PublisherTraceConfig.load(args.publisher_runtime);receiver=InboundAudioConfig.load(args.receiver_runtime)
    if not publisher.supports_external_rate_control:raise ValueError('explicit qualified rate-control publisher required')
    child=ROOT/'scripts/qualify_claire_session_worker.py';owner=ROOT/'scripts/qualify_claire_receiver_matrix.py'
    sources=[child,owner,Path(__file__)];before={str(p.relative_to(ROOT)):file_hash(p) for p in sources}
    args.directory.mkdir(parents=True,mode=0o700);rows=[]
    for kind in ('lifecycle','live'):
        path=args.directory/(kind+'.json');log_path=args.directory/(kind+'.log')
        command=[sys.executable,str(child),'--publisher-runtime',str(args.publisher_runtime.resolve()),'--trust-encoded-rate-control',
            '--receiver-runtime',str(args.receiver_runtime.resolve()),'--directory',str((args.directory/kind).resolve()),'--report',str(path.resolve())]
        if kind=='live':command.append('--live-qwen')
        for key in ('config','engine','scene'):command.extend(['--'+key,str(getattr(args,key).resolve())])
        environment=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        for key in ('LIVEKIT_LIB_PATH','LK_DISABLE_NVDEC','LD_PRELOAD','LD_AUDIT'):environment.pop(key,None)
        with open(log_path,'xb',opener=lambda p,f:os.open(p,f|os.O_NOFOLLOW,0o600)) as log:
            code=run_owned_process(command,environment,log,timeout=360)
        panic=any(word in log_path.read_bytes() for word in (b'panicked at',b'FFI Panic',b'non-unwinding panic'))
        stable=before=={str(p.relative_to(ROOT)):file_hash(p) for p in sources}
        receipt={'schema_version':1,'process_exit':code,'panic_detected':panic,'report_sha256':file_hash(path) if path.exists() else None,
            'log_sha256':file_hash(log_path),'qualifier_sha256':before[str(child.relative_to(ROOT))],
            'ownership_runner_sha256':before[str(owner.relative_to(ROOT))],'source_stable':stable}
        with open(args.directory/(kind+'-owner.json'),'x',opener=lambda p,f:os.open(p,f|os.O_NOFOLLOW,0o600)) as out:
            out.write(json.dumps(receipt,indent=2)+'\n')
        valid=code==0 and not panic and stable and path.exists() and path.stat().st_size<=4*1024*1024
        if valid:
            data=json.loads(path.read_text());binding=data['publisher_trace_runtime']
            valid=(data['source_sha256'].get(str(child.relative_to(ROOT)))==receipt['qualifier_sha256'] and
                data['publisher_stage_trace_enabled'] is False and data['source_capture_trace_enabled'] is False and
                data['publisher_probe_bitrate_bps'] is None and data['publisher_probe_rate_schedule'] is None and
                binding['descriptor_sha256']==publisher.revision and binding['trusted_external_rate_control'] is True and
                data['receiver_runtime']['library_sha256']==receiver.library_sha256 and len(data['cases'])==(19 if kind=='lifecycle' else 1))
        rows.append({'kind':kind,'diagnostic_completed':valid,**receipt})
        print(json.dumps({'kind':kind,'process_exit':code,'diagnostic_completed':valid}),flush=True)
        if not valid:break
    stable=(before=={str(p.relative_to(ROOT)):file_hash(p) for p in sources} and
        PublisherTraceConfig.load(args.publisher_runtime).revision==publisher.revision and InboundAudioConfig.load(args.receiver_runtime).revision==receiver.revision)
    completed=len(rows)==2 and all(x['diagnostic_completed'] for x in rows) and stable
    result={'schema_version':1,'status':'completed' if completed else 'failed','source_sha256':before,'cases':rows,
        'scope':'Owned process exits and exact report hashes for probes-disabled local SDK lifecycle/live checks; no browser or presentation acceptance.',
        'model_downloads':False,'software_downloads':False,'serving_configuration_changed':False}
    with open(args.report,'x',opener=lambda p,f:os.open(p,f|os.O_NOFOLLOW,0o600)) as out:out.write(json.dumps(result,indent=2)+'\n')
    return 0 if completed else 1


if __name__=='__main__':
    def interrupted(signum,frame):raise KeyboardInterrupt('qualification interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('publisher-runtime','receiver-runtime','config','engine','scene','directory','report'):parser.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(main(parser.parse_args()))
