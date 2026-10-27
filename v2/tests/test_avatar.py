from dataclasses import replace
from fractions import Fraction
from importlib.resources import files
import json
import math
from pathlib import Path
import shutil

import pytest

from talkinghead_v2.assets.avatar import digest, load_avatar
from talkinghead_v2.assets.artifacts import sha256_file
from talkinghead_v2.contracts import ContractError, Envelope, GenerationKey, MotionFrame
from talkinghead_v2.rendering.mesh import MeshRenderer, deform
from talkinghead_v2.rendering.timeline import FrameMailbox, MotionTimeline
from talkinghead_v2.retarget import RetargetProfile, compose_behavior, fit_landmarks, idle_coefficients


@pytest.fixture
def asset():
    return load_avatar(Path(str(files('talkinghead_v2')/'fixtures/p03-avatar/avatar.json')),tenant_id='procedural')


def frame(asset, sample=0, sequence=0, **kwargs):
    return MotionFrame(Envelope(GenerationKey('s','t',0),sequence,'test'),asset.rig,sample,48000,sample+1600,
                       kwargs.pop('coefficients',(0.0,0.0)),**kwargs)


@pytest.mark.parametrize('mutation', ['tenant','approval','use','hash','topology','basis','camera','representation','consent'])
def test_manifest_denies_unauthorized_or_incompatible_assets_before_loading(tmp_path,mutation):
    source=Path(str(files('talkinghead_v2')/'fixtures/p03-avatar'))
    shutil.copytree(source,tmp_path/'asset')
    path=tmp_path/'asset/avatar.json'
    data=json.loads(path.read_text())
    if mutation=='tenant': data['tenant_id']='elsewhere'
    elif mutation=='approval': data['status']='revoked'
    elif mutation=='use': data['allowed_uses']=[]
    elif mutation=='hash': data['appearance']['sha256']='sha256:'+'0'*64
    elif mutation in ('topology','basis'): data['rig'][mutation+'_hash']='sha256:'+'0'*64
    elif mutation=='camera': data['camera']['world_to_camera_rotation'][0][0]=-1
    elif mutation=='representation': data['appearance']['representation']='attached_3dgs'
    else:
        data['kind']='human_consented'; data['allowed_uses']=['research_avatar']
    path.write_text(json.dumps(data))
    with pytest.raises(ContractError): load_avatar(path,tenant_id='procedural')


def test_mesh_hashes_cover_neutral_geometry_topology_and_jaw(asset,tmp_path):
    assert len(asset.vertices)==8 and len(asset.faces)==8
    assert len(asset.rig.basis_hash)==71
    bad=replace(frame(asset),rig=replace(asset.rig,basis_hash='sha256:'+'0'*64))
    with pytest.raises(ContractError,match='mismatch'): deform(asset,bad)
    with pytest.raises(ContractError,match='bounds'): deform(asset,frame(asset,coefficients=(1.1,0)))


@pytest.mark.parametrize('mutation',['face','vertex_count','jaw','symlink'])
def test_mesh_refuses_coercion_even_when_file_hash_is_updated(tmp_path,mutation):
    source=Path(str(files('talkinghead_v2')/'fixtures/p03-avatar'))
    shutil.copytree(source,tmp_path/'asset')
    path=tmp_path/'asset/avatar.json'
    mesh_path=path.parent/'mesh.json'
    manifest=json.loads(path.read_text());mesh=json.loads(mesh_path.read_text())
    if mutation=='face': mesh['faces'][0][0]=len(mesh['vertices'])
    elif mutation=='vertex_count': mesh['blendshapes']['lip_close'].pop()
    elif mutation=='jaw': mesh['jaw_vertices'].append(len(mesh['vertices']))
    else:
        mesh_path.rename(path.parent/'original.json')
        mesh_path.symlink_to('original.json')
    if mutation!='symlink': mesh_path.write_text(json.dumps(mesh))
    manifest['appearance']['sha256']='sha256:'+sha256_file(mesh_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ContractError): load_avatar(path,tenant_id='procedural')


def test_rigid_jaw_rotation_does_not_translate_whole_head_or_coerce_channels(asset):
    neutral=deform(asset,frame(asset))
    angle=.3
    moved=deform(asset,frame(asset,jaw_xyzw=(math.sin(angle/2),0,0,math.cos(angle/2))))
    assert moved[0]==neutral[0] and moved[1]==neutral[1]
    assert moved[2] != neutral[2]
    for i in asset.jaw_vertices:
        assert sum(x*x for x in moved[i])==pytest.approx(sum(x*x for x in neutral[i]))


def test_explicit_retarget_and_bounded_landmark_fit(asset):
    source=replace(asset.rig,schema_id='calibration-source',basis_hash='sha256:'+'3'*64)
    profile=RetargetProfile(source,asset.rig,((1.,0.),(0.,1.)),(0.,0.),asset.bounds,digest({'fixture':'identity'}))
    original=replace(frame(asset,coefficients=(.65,.2)),rig=source)
    result=profile.apply(original)
    assert result.rig==asset.rig and result.coefficients==(.65,.2)
    with pytest.raises(ContractError,match='calibration'): profile.apply(replace(original,rig=asset.rig))
    indices=tuple(range(len(asset.vertices)))
    target=deform(asset,result)
    fitted=fit_landmarks(asset,indices,target)
    assert fitted['coefficients']==pytest.approx((.65,.2),abs=1e-4)
    assert fitted['landmark_rmse_m']<1e-6 and not fitted['jaw_fitted']
    impossible=tuple((x,y+100,z) for x,y,z in target)
    bounded=fit_landmarks(asset,indices,impossible)
    assert all(0<=v<=1 for v in bounded['coefficients'])
    assert bounded['landmark_rmse_m']>1


def test_behavior_protects_articulation_and_cancellation_is_neutral(asset):
    values=idle_coefficients(asset,4800,48000,mode='listening',blink_channel='upper_marker',protected=('lip_close',))
    assert values==(0.,1.)
    with pytest.raises(ContractError,match='protected'):
        idle_coefficients(asset,4800,48000,mode='speaking',blink_channel='lip_close',protected=('lip_close',))
    assert idle_coefficients(asset,4800,48000,mode='cancelled',blink_channel='upper_marker')==(0.,0.)
    with pytest.raises(ContractError,match='protected'):
        idle_coefficients(asset,4800,48000,mode='speaking',blink_channel='lip_close')
    speech=frame(asset,4800,coefficients=(.65,0.))
    composed=compose_behavior(asset,speech,mode='speaking',blink_channel='upper_marker')
    assert composed.coefficients==(.65,1.) and composed.jaw_xyzw==speech.jaw_xyzw


def test_mesh_pixels_and_metadata_change_with_valid_articulation(asset):
    renderer=MeshRenderer(asset,resolution=64)
    opened=renderer.render(frame(asset),duration=Fraction(1,30))
    closed=renderer.render(frame(asset,coefficients=(1.,0.)),duration=Fraction(1,30))
    assert len(opened.rgb)==64*64*3 and opened.rgb!=closed.rgb
    assert opened.duration==Fraction(1,30) and opened.pts_samples==0
    assert 'no device presentation' in opened.boundary


def test_motion_cadence_independent_interpolation_and_epoch_fence(asset):
    timeline=MotionTimeline(frame(asset).envelope.key,asset.rig,48000)
    assert timeline.push(frame(asset))
    assert timeline.push(frame(asset,960,1,coefficients=(1.,0.)))
    assert timeline.select(480).coefficients==(.5,0.)
    held=timeline.select(960)
    timeline.cancel()
    assert timeline.select(1000) is None
    assert not timeline.push(held) and timeline.rejected_stale==1


def test_motion_queue_bounds_time_items_order_and_validity(asset):
    key=frame(asset).envelope.key
    timeline=MotionTimeline(key,asset.rig,48000,max_items=2)
    timeline.push(frame(asset));timeline.push(frame(asset,960,1))
    with pytest.raises(ContractError,match='backpressure'): timeline.push(frame(asset,1920,2))
    timeline.select(960)
    with pytest.raises(ContractError,match='reordered'): timeline.push(frame(asset,960,2))
    with pytest.raises(ContractError,match='backpressure'): timeline.push(frame(asset,30000,2))
    assert timeline.select(5000) is None
    with pytest.raises(ContractError,match='monotonic'): timeline.select(0)


def test_thirty_virtual_minutes_of_motion_have_bounded_history(asset):
    timeline=MotionTimeline(frame(asset).envelope.key,asset.rig,48000)
    for index in range(1800*60):
        sample=index*800
        timeline.advance(sample)
        timeline.push(frame(asset,sample,index))
        assert timeline.select(sample) is not None
    assert timeline.peak_items<=2
    assert len(timeline.frames)<=2


def test_completed_video_is_fenced_after_cancel_and_backpressure_has_both_bounds(asset):
    image=MeshRenderer(asset,resolution=16).render(frame(asset),duration=Fraction(1,30))
    mailbox=FrameMailbox(image.key,max_items=1)
    assert mailbox.submit(image,playhead=0)
    with pytest.raises(ContractError,match='item bound'):
        mailbox.submit(replace(image,pts_samples=1600),playhead=0)
    assert mailbox.take_due(playhead=0)==image
    with pytest.raises(ContractError,match='future time'):
        mailbox.submit(replace(image,pts_samples=4800),playhead=0)
    mailbox.cancel()
    assert not mailbox.submit(replace(image,pts_samples=1600),playhead=0)
    assert mailbox.take_due(playhead=1600) is None


def test_late_video_is_dropped_without_shifting_audio_pts(asset):
    image=MeshRenderer(asset,resolution=16).render(frame(asset),duration=Fraction(1,30))
    mailbox=FrameMailbox(image.key)
    assert not mailbox.submit(image,playhead=2000)
    assert mailbox.dropped_late==1
    assert mailbox.submit(replace(image,pts_samples=3200),playhead=2000)
    assert mailbox.take_due(playhead=3199) is None
    assert mailbox.take_due(playhead=3200).pts_samples==3200
    with pytest.raises(ContractError,match='monotonic'): mailbox.take_due(playhead=0)


def test_offline_mesh_replay_contains_real_pixels_audio_and_explicit_limits(tmp_path):
    from types import SimpleNamespace
    from talkinghead_v2.avatar_replay import run
    args=SimpleNamespace(output=tmp_path/'result',resolution=16,wave=None,tts_report=None,case='short')
    report=run(args)
    assert report['render']['frames']==90 and report['render']['completed_stale_callbacks_rejected']
    assert not report['actual_device_presentation'] and not report['neural_motion_inference']
    assert (args.output/'video.rgb').stat().st_size==90*16*16*3
    assert report['claire_frontend']['model_ran'] is False
    assert report['claire_frontend']['peak_buffered_samples']<=8640
    assert report['claire_frontend']['context_windows']==180
    assert all(row['logical_skew_samples']==0 for row in report['render']['frame_trace'])
    with pytest.raises(ContractError,match='already exists'): run(args)
