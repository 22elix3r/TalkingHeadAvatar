"""Serving policy consumes pinned evidence without loading a native library."""
import copy,json,hashlib
from pathlib import Path
import pytest
from talkinghead_v2.contracts import ContractError
from talkinghead_v2.publish_probe import RATE_PATCH_SHA256
from talkinghead_v2.server.inbound_config import RUST_REVISION,CARGO_LOCK_SHA256


def evidence(tmp_path):
    hashes={role:letter*64 for role,letter in [('executable','a'),('ffi_library','b'),('livekit_library','c'),
        ('bridge_library','d'),('ptx','e'),('engine','f'),('scene','1')]}
    paths={role:tmp_path/role for role in hashes}
    binding={'executable_sha256':hashes['executable'],'ffi_sha256':hashes['ffi_library'],'cpp_sha256':hashes['livekit_library'],
        'descriptor_sha256':'2'*64,'trusted_external_rate_control':True,'serving_runtime':False}
    sender={'valid_samples':4800,'source_video_frames':3,'video_skipped':0}
    receiver={'video':[{'source_sample':i} for i in (0,1600,3200)],'missing_source_frame_indices':[]}
    base={'protocol':'THA_CLAIRE_MEDIA_2','eligible_motion_start_buffer_ms':500,'motion_input_lead_bound_ms':800,
        'publisher_trace_runtime':binding,'publisher_stage_trace_enabled':False,'source_capture_trace_enabled':False,
        'publisher_probe_bitrate_bps':None,'publisher_probe_rate_schedule':None,'fresh_publication_ids':True,'admission_reacquired':True,
        'source_sha256':{'scripts/qualify_claire_session_worker.py':'3'*64},
        **{field:hashes[role] for field,role in [('binary_sha256','executable'),('bridge_sha256','bridge_library'),('ptx_sha256','ptx'),('engine_sha256','engine'),('scene_sha256','scene')]}}
    names=('qwen_recorded','segmented_gap','one','short','aligned','partial','quantum','tail','grid_before','grid_exact','grid_after','grid_second','grid_third')
    lifecycle={**copy.deepcopy(base),'live_qwen_streaming':False,'cases':[{'name':name,'process_exit':0,'sender':copy.deepcopy(sender)} for name in names]}
    lifecycle['cases'][0]['receiver']=copy.deepcopy(receiver)
    lifecycle['cases'] += [{'name':name,'rejected':True,'process_exit':1} for name in ('empty','wrong_epoch','gap','sequence','wrong_motion')]
    lifecycle['cases'].append({'name':'backpressure_cancel','blocked_producer':True,'owned_process_exit_ms':100.,'receiver_flush_claimed':False})
    live={**copy.deepcopy(base),'live_qwen_streaming':True,'cases':[{'name':'qwen_live','process_exit':0,'sender':copy.deepcopy(sender),'receiver':copy.deepcopy(receiver)}]}
    build={'rust_revision':RUST_REVISION,'cargo_lock_sha256':CARGO_LOCK_SHA256,'patch_sha256':RATE_PATCH_SHA256,
        'library_sha256':hashes['ffi_library'],'external_rate_control_opt_in':True,'trace_tests_passed':5,'offline':True}
    row={'process_exit':0,'panic_detected':False,'diagnostic_completed':True,'publisher_runtime':copy.deepcopy(binding),
        'source_frames':3,'captured_source_frames':3,'received_source_frames':3,'missing_at_encoder_output':[],
        'missing_at_packetizer':[],'missing_at_receiver':[],'feedback':{'schedule':'down-up'}}
    data={'publisher_build_report':build,'publisher_lifecycle_report':lifecycle,'publisher_live_report':live,
        'publisher_lifecycle_owner':{'process_exit':0,'panic_detected':False},'publisher_live_owner':{'process_exit':0,'panic_detected':False},
        'publisher_rate_report':{'status':'completed','encoder_ceiling_schedule':'down-up','cases':[copy.deepcopy(row) for _ in range(3)]}}
    def save(rebind=True):
        for role,value in data.items():
            if role.endswith('_owner'):continue
            paths[role]=tmp_path/(role+'.json');paths[role].write_text(json.dumps(value));paths[role].chmod(0o600)
            hashes[role]=hashlib.sha256(paths[role].read_bytes()).hexdigest()
        for kind in ('lifecycle','live'):
            role='publisher_'+kind+'_owner'
            if rebind:data[role]['report_sha256']=hashes['publisher_'+kind+'_report']
            data[role].setdefault('source_stable',True);data[role].setdefault('qualifier_sha256','3'*64)
            paths[role]=tmp_path/(role+'.json');paths[role].write_text(json.dumps(data[role]));paths[role].chmod(0o600)
    save();return paths,hashes,data,save


def check(mode,paths,hashes):
    from talkinghead_v2.server.claire_publisher import validate_publisher_policy
    return validate_publisher_policy(mode,paths,hashes,500)


def test_default_policy_never_enables_inherited_trust(tmp_path):
    assert check('sdk_default',{}, {}) is False
    with pytest.raises(ContractError):check('external_trusted_local_research',{}, {})
    with pytest.raises(ContractError):check('trust_everything',{}, {})


def test_explicit_policy_requires_matching_lifecycle_live_rate_and_build_evidence(tmp_path):
    paths,hashes,_,_=evidence(tmp_path)
    assert check('external_trusted_local_research',paths,hashes) is True


def test_clean_owner_receipt_from_another_report_cannot_authorize_post_report_failure(tmp_path):
    paths,hashes,data,save=evidence(tmp_path)
    data['publisher_live_owner']['report_sha256']=data['publisher_lifecycle_owner']['report_sha256']
    save(rebind=False)
    with pytest.raises(ContractError):check('external_trusted_local_research',paths,hashes)


def test_missing_source_identity_in_both_owner_and_report_is_not_a_match(tmp_path):
    paths,hashes,data,save=evidence(tmp_path)
    data['publisher_live_report']['source_sha256']={}
    data['publisher_live_owner']['qualifier_sha256']=None
    save()
    with pytest.raises(ContractError):check('external_trusted_local_research',paths,hashes)


@pytest.mark.parametrize('change',[
    'wrong_ffi','wrong_worker','wrong_engine','probes_enabled','receiver_gap','partial_receiver','failed_live_owner',
    'panic_after_report','missing_fault','fault_accepted','not_cancelled','wrong_patch','untested_sdk','untrusted_rate_turn',
    'incomplete_rate_matrix','different_rate_runtime','different_live_runtime','different_library_directory',
])
def test_unqualified_or_mixed_evidence_cannot_enable_serving_policy(tmp_path,change):
    paths,hashes,data,save=evidence(tmp_path)
    if change=='wrong_ffi':data['publisher_build_report']['library_sha256']='0'*64
    elif change=='wrong_worker':data['publisher_live_report']['binary_sha256']='0'*64
    elif change=='wrong_engine':data['publisher_live_report']['engine_sha256']='0'*64
    elif change=='probes_enabled':data['publisher_live_report']['publisher_probe_rate_schedule']='down-up'
    elif change=='receiver_gap':data['publisher_live_report']['cases'][0]['receiver']['missing_source_frame_indices']=[1]
    elif change=='partial_receiver':data['publisher_live_report']['cases'][0]['receiver']['video'].pop()
    elif change=='failed_live_owner':data['publisher_live_owner']['process_exit']=1
    elif change=='panic_after_report':data['publisher_lifecycle_owner']['panic_detected']=True
    elif change=='missing_fault':data['publisher_lifecycle_report']['cases'].pop(13)
    elif change=='fault_accepted':data['publisher_lifecycle_report']['cases'][13]['process_exit']=0
    elif change=='not_cancelled':data['publisher_lifecycle_report']['cases'][-1]['blocked_producer']=False
    elif change=='wrong_patch':data['publisher_build_report']['patch_sha256']='0'*64
    elif change=='untested_sdk':data['publisher_build_report']['trace_tests_passed']=0
    elif change=='untrusted_rate_turn':data['publisher_rate_report']['cases'][0]['publisher_runtime']['trusted_external_rate_control']=False
    elif change=='incomplete_rate_matrix':data['publisher_rate_report']['cases'].pop()
    elif change=='different_rate_runtime':data['publisher_rate_report']['cases'][0]['publisher_runtime']['descriptor_sha256']='0'*64
    elif change=='different_live_runtime':data['publisher_live_report']['publisher_trace_runtime']['ffi_sha256']='0'*64
    elif change=='different_library_directory':paths['ffi_library']=tmp_path/'other/ffi.so'
    save()
    with pytest.raises(ContractError):check('external_trusted_local_research',paths,hashes)
