"""Evidence-bound, local research selection for the pre-encoded SDK policy."""
import math
import re
from talkinghead_v2.contracts import ContractError
from talkinghead_v2.publish_probe import RATE_PATCH_SHA256
from talkinghead_v2.server.inbound_config import private_json,RUST_REVISION,CARGO_LOCK_SHA256

PUBLISHER_ROLES=frozenset(('publisher_build_report','publisher_lifecycle_report','publisher_live_report',
    'publisher_lifecycle_owner','publisher_live_owner','publisher_rate_report'))
NORMAL_CASES=frozenset(('qwen_recorded','segmented_gap','one','short','aligned','partial','quantum','tail',
    'grid_before','grid_exact','grid_after','grid_second','grid_third'))
FAULT_CASES=frozenset(('empty','wrong_epoch','gap','sequence','wrong_motion'))


def require(value,reason):
    if not value:raise ContractError('unqualified Claire publisher: '+reason)


def _receiver_complete(case):
    sender=case['sender'];receiver=case['receiver'];count=sender['valid_samples']
    require(type(count) is int and 0<count<=48000*120,'bounded source samples')
    expected=(count+1599)//1600
    require(sender['source_video_frames']==expected and sender['video_skipped']==0 and
        receiver['missing_source_frame_indices']==[] and
        [row['source_sample'] for row in receiver['video']]==list(range(0,count,1600)),
        'complete source-frame receiver evidence')


def validate_publisher_policy(mode,paths,hashes,prebuffer_ms):
    if mode=='sdk_default':return False
    require(mode=='external_trusted_local_research','explicit local research policy')
    require(PUBLISHER_ROLES<=paths.keys(),'complete evidence inventory')
    require(len({paths[role].parent for role in ('executable','ffi_library','livekit_library','bridge_library')})==1,
        'one immutable native library directory')
    evidence={role:private_json(paths[role],4*1024*1024) for role in PUBLISHER_ROLES}
    try:
        build=evidence['publisher_build_report']
        require(build.get('rust_revision')==RUST_REVISION and build.get('cargo_lock_sha256')==CARGO_LOCK_SHA256 and
            build.get('patch_sha256')==RATE_PATCH_SHA256 and build.get('library_sha256')==hashes['ffi_library'] and
            build.get('external_rate_control_opt_in') is True and type(build.get('trace_tests_passed')) is int and
            build['trace_tests_passed']==5 and build.get('offline') is True,'tested exact SDK build')
        binding=None
        for kind in ('lifecycle','live'):
            report=evidence['publisher_'+kind+'_report'];current=report['publisher_trace_runtime']
            owner=evidence['publisher_'+kind+'_owner']
            qualifier=owner.get('qualifier_sha256')
            require(type(owner.get('process_exit')) is int and owner['process_exit']==0 and owner.get('panic_detected') is False and
                owner.get('report_sha256')==hashes['publisher_'+kind+'_report'] and owner.get('source_stable') is True and
                isinstance(qualifier,str) and re.fullmatch('[a-f0-9]{64}',qualifier) and
                qualifier==report.get('source_sha256',{}).get('scripts/qualify_claire_session_worker.py'),
                'clean owner exit bound to this exact report and qualifier')
            require(current.get('trusted_external_rate_control') is True and current.get('serving_runtime') is False and
                current.get('executable_sha256')==hashes['executable'] and current.get('ffi_sha256')==hashes['ffi_library'] and
                current.get('cpp_sha256')==hashes['livekit_library'],'measured publisher runtime binding')
            if binding is None:binding=current
            require(current==binding,'one qualification publisher inventory')
            require(report.get('protocol')=='THA_CLAIRE_MEDIA_2' and report.get('eligible_motion_start_buffer_ms')==prebuffer_ms and
                report.get('motion_input_lead_bound_ms')==800 and report.get('publisher_stage_trace_enabled') is False and
                report.get('source_capture_trace_enabled') is False and report.get('publisher_probe_bitrate_bps') is None and
                report.get('publisher_probe_rate_schedule') is None and report.get('fresh_publication_ids') is True and
                report.get('admission_reacquired') is True,'probes-disabled lifecycle policy')
            for field,role in [('binary_sha256','executable'),('bridge_sha256','bridge_library'),('ptx_sha256','ptx'),
                ('engine_sha256','engine'),('scene_sha256','scene')]:
                require(report.get(field)==hashes[role],'worker/model/scene binding')
            cases=report['cases'];require(isinstance(cases,list),'bounded qualification cases')
            if kind=='live':
                require(report.get('live_qwen_streaming') is True and len(cases)==1 and cases[0].get('name')=='qwen_live' and
                    cases[0].get('process_exit')==0,'actual live-Qwen qualification')
                _receiver_complete(cases[0])
            else:
                require(report.get('live_qwen_streaming') is False and len(cases)==19 and
                    {c['name'] for c in cases}==NORMAL_CASES|FAULT_CASES|{'backpressure_cancel'},'full lifecycle cohort')
                for case in cases:
                    if case['name'] in NORMAL_CASES:require(case.get('process_exit')==0,'normal EOF exit')
                    elif case['name'] in FAULT_CASES:require(case.get('rejected') is True and case.get('process_exit')==1,'rejected protocol fault')
                    else:
                        elapsed=case.get('owned_process_exit_ms')
                        require(case.get('blocked_producer') is True and case.get('receiver_flush_claimed') is False and
                            type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>0,'owned backpressure cancellation')
                _receiver_complete(next(c for c in cases if c['name']=='qwen_recorded'))
        rate=evidence['publisher_rate_report'];rows=rate['cases']
        require(rate.get('status')=='completed' and rate.get('encoder_ceiling_schedule')=='down-up' and
            isinstance(rows,list) and len(rows)==3,'in-turn encoder-control study')
        for row in rows:
            count=row.get('source_frames')
            require(type(count) is int and count>0 and row.get('captured_source_frames')==count and row.get('received_source_frames')==count and
                row.get('missing_at_encoder_output')==[] and row.get('missing_at_packetizer')==[] and row.get('missing_at_receiver')==[] and
                row.get('process_exit')==0 and row.get('panic_detected') is False and row.get('diagnostic_completed') is True and
                row.get('publisher_runtime')==binding and row.get('feedback',{}).get('schedule')=='down-up',
                'complete rate-study runtime/frames')
    except (KeyError,TypeError,AttributeError,ValueError) as error:
        if isinstance(error,ContractError):raise
        raise ContractError('unqualified Claire publisher: malformed evidence') from None
    return True
