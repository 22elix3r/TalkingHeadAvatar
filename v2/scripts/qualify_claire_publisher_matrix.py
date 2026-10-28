"""Matched per-frame publish-stage comparison; no model/software fetching."""
import argparse,json,os,sys,signal
from pathlib import Path
from qualify_claire_receiver_matrix import run_owned_process
from talkinghead_v2.publish_probe import PublisherTraceConfig
from talkinghead_v2.server.inbound_config import file_hash,InboundAudioConfig

ROOT=Path(__file__).resolve().parents[1]


def main(args):
    if args.directory.exists() or args.report.exists():raise ValueError('preserve previous publisher evidence')
    publisher=PublisherTraceConfig.load(args.publisher_runtime)
    receiver=InboundAudioConfig.load(args.receiver_runtime)
    if not publisher.supports_external_rate_control:raise ValueError('explicit opt-in-capable publisher probe required')
    child=ROOT/'scripts/qualify_claire_session_worker.py';child_sha=file_hash(child);runner_sha=file_hash(Path(__file__))
    owner_sha=file_hash(ROOT/'scripts/qualify_claire_receiver_matrix.py')
    args.directory.mkdir(parents=True,mode=0o700);cases=[]
    for index in range(1,args.repeats+1):
        for trusted in (False,True):
            name=('external-trusted' if trusted else 'sdk-default')+'-'+str(index)
            report_path=args.directory/(name+'.json');log_path=args.directory/(name+'.log')
            command=[sys.executable,str(child),'--live-qwen','--trace-source-captures','--trace-receiver-ffi',
                '--receiver-runtime',str(args.receiver_runtime.resolve()),'--native-video-queue-frames','1',
                '--publisher-trace-runtime',str(args.publisher_runtime.resolve()),'--directory',str((args.directory/name).resolve()),'--report',str(report_path.resolve())]
            if trusted:command.append('--trust-encoded-rate-control')
            for key in ('config','engine','scene'):command.extend(['--'+key,str(getattr(args,key).resolve())])
            environment=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            for key in ('LIVEKIT_LIB_PATH','LK_DISABLE_NVDEC','THA_LK_TRUST_EXTERNAL_RATE_CONTROL','LD_PRELOAD','LD_AUDIT'):environment.pop(key,None)
            with open(log_path,'xb',opener=lambda p,f:os.open(p,f|os.O_NOFOLLOW,0o600)) as log:
                code=run_owned_process(command,environment,log)
            data=log_path.read_bytes();panic=any(word in data for word in (b'panicked at',b'FFI Panic',b'non-unwinding panic'))
            row={'name':name,'trusted_external_rate_control':trusted,'process_exit':code,'panic_detected':panic,'log_sha256':file_hash(log_path)}
            valid=code==0 and not panic and report_path.exists() and report_path.stat().st_size<=4*1024*1024
            if valid:
                observed=json.loads(report_path.read_text());case=observed['cases'][0];trace=case['publisher_stage_trace'];r=case['receiver']
                expected=(case['sender']['valid_samples']+1599)//1600
                capture=case['source_capture_trace'];source_by_id={x['frame_id']:x['source_sample']//1600 for x in capture}
                row.update({'source_report_sha256':file_hash(report_path),'source_frames':expected,'captured_source_frames':len(capture),
                    'source_frame_ids_by_stage':trace['source_frame_ids_by_stage'],
                    'missing_source_indices_at_encoder_output':[source_by_id[x] for x in trace['missing_source_frame_ids_at_encoder_output']],
                    'missing_source_indices_at_packetizer':[source_by_id[x] for x in trace['missing_source_frame_ids_at_packetizer']],
                    'missing_source_indices_at_receiver':r['missing_source_frame_indices'],
                    'received_source_frames':len(r['video']),'ffi_callbacks_missing_at_consumer':r['ffi_video_boundary']['ffi_callbacks_missing_at_consumer'],
                    'non_source_or_unidentified_events_by_stage':trace['non_source_or_unidentified_events_by_stage'],
                    'source_payload_average_bps':sum(x['encoded_bytes'] for x in capture)*8/(case['sender']['valid_samples']/48000),
                    'sender':case['sender'],'sender_rtp_stats':case['sender_rtp_stats'],'receiver_video_rtp_stats':r['video_rtp_stats'],
                    'publisher_runtime':observed['publisher_trace_runtime'],'receiver_runtime':observed['receiver_runtime']})
                valid=(observed['source_sha256']['scripts/qualify_claire_session_worker.py']==child_sha and
                    observed['publisher_trace_runtime']['descriptor_sha256']==publisher.revision and
                    observed['publisher_trace_runtime']['trusted_external_rate_control'] is trusted and
                    observed['receiver_runtime']['library_sha256']==receiver.library_sha256)
                row['all_source_frames_at_all_observed_boundaries']=(len(capture)==expected and not trace['missing_source_frame_ids_at_encoder_output'] and
                    not trace['missing_source_frame_ids_at_packetizer'] and not r['missing_source_frame_indices'] and not r['ffi_video_boundary']['ffi_callbacks_missing_at_consumer'])
            row['diagnostic_completed']=valid;cases.append(row)
            print(json.dumps({k:row.get(k) for k in ('name','process_exit','diagnostic_completed','received_source_frames','missing_source_indices_at_encoder_output')}),flush=True)
            if not valid:break
        if not cases[-1]['diagnostic_completed']:break
    stable=(file_hash(child)==child_sha and file_hash(Path(__file__))==runner_sha and file_hash(ROOT/'scripts/qualify_claire_receiver_matrix.py')==owner_sha and
        PublisherTraceConfig.load(args.publisher_runtime).revision==publisher.revision and InboundAudioConfig.load(args.receiver_runtime).revision==receiver.revision)
    completed=len(cases)==2*args.repeats and all(x['diagnostic_completed'] for x in cases) and stable
    trusted=[x for x in cases if x['trusted_external_rate_control']]
    report={'schema_version':1,'status':'completed' if completed else 'failed','cases':cases,
        'qualifier_sha256':child_sha,'runner_sha256':runner_sha,'ownership_runner_sha256':owner_sha,
        'trusted_cases_conserve_all_source_frames':completed and all(x['all_source_frames_at_all_observed_boundaries'] for x in trusted),
        'scope':'Alternating same-host fixture and explicit SDK flag. Transformer hooks are not NVENC execution or network/physical delivery acknowledgments. No stable loss-rate, rate-control under constrained network, latency, human quality or release claim.',
        'model_downloads':False,'software_downloads':False,'serving_configuration_changed':False}
    args.report.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with open(args.report,'x',opener=lambda p,f:os.open(p,f|os.O_NOFOLLOW,0o600)) as output:output.write(json.dumps(report,indent=2)+'\n')
    return 0 if completed else 1


if __name__=='__main__':
    def interrupted(signum,frame):raise KeyboardInterrupt('qualification interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repeats',type=int,choices=(1,3),default=3)
    for name in ('publisher-runtime','receiver-runtime','config','engine','scene','directory','report'):parser.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(main(parser.parse_args()))
