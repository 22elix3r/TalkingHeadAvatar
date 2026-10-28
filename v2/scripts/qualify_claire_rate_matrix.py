"""Isolated fixed-ceiling feedback observations; no WAN/congestion qualification."""
import argparse,json,os,signal,sys
from pathlib import Path
from qualify_claire_receiver_matrix import run_owned_process
from talkinghead_v2.publish_probe import PublisherTraceConfig
from talkinghead_v2.server.inbound_config import InboundAudioConfig,file_hash

ROOT=Path(__file__).resolve().parents[1]


def main(args):
    if args.directory.exists() or args.report.exists():raise ValueError('preserve earlier rate observations')
    publisher=PublisherTraceConfig.load(args.publisher_runtime);receiver=InboundAudioConfig.load(args.receiver_runtime)
    if not publisher.supports_external_rate_control:raise ValueError('explicit rate-control-capable publisher required')
    sources=[Path(__file__),ROOT/'scripts/qualify_claire_session_worker.py',ROOT/'scripts/qualify_claire_receiver_matrix.py']
    before={str(p.relative_to(ROOT)):file_hash(p) for p in sources}
    args.directory.mkdir(parents=True,mode=0o700);rows=[]
    for ceiling in (300000,1200000,600000,2400000):
        name='ceiling-'+str(ceiling);report_path=args.directory/(name+'.json');log_path=args.directory/(name+'.log')
        command=[sys.executable,str(sources[1]),'--live-qwen','--trace-source-captures','--trace-receiver-ffi',
            '--publisher-trace-runtime',str(args.publisher_runtime.resolve()),'--trust-encoded-rate-control',
            '--probe-bitrate-bps',str(ceiling),'--receiver-runtime',str(args.receiver_runtime.resolve()),
            '--directory',str((args.directory/name).resolve()),'--report',str(report_path.resolve())]
        for key in ('config','engine','scene'):command.extend(['--'+key,str(getattr(args,key).resolve())])
        environment=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        for key in ('LIVEKIT_LIB_PATH','LK_DISABLE_NVDEC','LD_PRELOAD','LD_AUDIT'):environment.pop(key,None)
        with open(log_path,'xb',opener=lambda p,f:os.open(p,f|os.O_NOFOLLOW,0o600)) as log:
            code=run_owned_process(command,environment,log)
        panic=any(word in log_path.read_bytes() for word in (b'panicked at',b'FFI Panic',b'non-unwinding panic'))
        row={'ceiling_bps':ceiling,'process_exit':code,'panic_detected':panic,'log_sha256':file_hash(log_path)}
        valid=code==0 and not panic and report_path.exists() and report_path.stat().st_size<=4*1024*1024
        if valid:
            data=json.loads(report_path.read_text());case=data['cases'][0];capture=case['source_capture_trace']
            count=case['sender']['valid_samples'];expected=(count+1599)//1600;trace=case['publisher_stage_trace']
            windows=[]
            for start in range(0,count,48000):
                end=min(count,start+48000);bits=sum(x['encoded_bytes']*8 for x in capture if start<=x['source_sample']<end)
                windows.append({'start_sample':start,'end_sample_exclusive':end,'payload_bits':bits,'average_bps':bits*48000/(end-start)})
            row.update(source_report_sha256=file_hash(report_path),source_frames=expected,captured_source_frames=len(capture),
                received_source_frames=len(case['receiver']['video']),feedback=case['publisher_rate_probe'],
                source_payload_average_bps=sum(x['encoded_bytes'] for x in capture)*8*48000/count,
                source_payload_windows_48khz=windows,largest_source_access_unit_bytes=max(x['encoded_bytes'] for x in capture),
                missing_at_encoder_output=trace['missing_source_frame_ids_at_encoder_output'],
                missing_at_packetizer=trace['missing_source_frame_ids_at_packetizer'],
                missing_at_receiver=case['receiver']['missing_source_frame_indices'],sender=case['sender'],
                publisher_runtime=data['publisher_trace_runtime'],receiver_runtime=data['receiver_runtime'])
            valid=(data['source_sha256']['scripts/qualify_claire_session_worker.py']==before['scripts/qualify_claire_session_worker.py'] and
                data['publisher_trace_runtime']['descriptor_sha256']==publisher.revision and data['publisher_probe_bitrate_bps']==ceiling and
                data['publisher_trace_runtime']['trusted_external_rate_control'] is True and data['publisher_stage_trace_enabled'] is True and
                data['receiver_runtime']['library_sha256']==receiver.library_sha256)
        row['diagnostic_completed']=valid;rows.append(row)
        print(json.dumps({k:row.get(k) for k in ('ceiling_bps','process_exit','diagnostic_completed','source_payload_average_bps','received_source_frames')}),flush=True)
        if not valid:break
    stable=(before=={str(p.relative_to(ROOT)):file_hash(p) for p in sources} and
        PublisherTraceConfig.load(args.publisher_runtime).revision==publisher.revision and InboundAudioConfig.load(args.receiver_runtime).revision==receiver.revision)
    completed=len(rows)==4 and all(x['diagnostic_completed'] for x in rows) and stable
    result={'schema_version':1,'status':'completed' if completed else 'failed','source_sha256':before,'cases':rows,
        'scope':'Four separate fixed publish/encoder ceilings, one local SFU. Payload averages exclude RTP overhead; partial final windows remain explicit. No in-turn bandwidth transition, network shaping, pacer delay, physical presentation, quality or rate-control acceptance.',
        'model_downloads':False,'software_downloads':False,'serving_configuration_changed':False}
    args.report.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with open(args.report,'x',opener=lambda p,f:os.open(p,f|os.O_NOFOLLOW,0o600)) as output:output.write(json.dumps(result,indent=2)+'\n')
    return 0 if completed else 1


if __name__=='__main__':
    def interrupted(signum,frame):raise KeyboardInterrupt('qualification interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('publisher-runtime','receiver-runtime','config','engine','scene','directory','report'):parser.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(main(parser.parse_args()))
