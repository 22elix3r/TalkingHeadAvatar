"""Real PostgreSQL lifecycle and CPU importer, with serving providers explicit fakes."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from threading import Barrier

import pytest
from sqlalchemy import select

from test_control_api import settings,client
from talkinghead_v2.server.db import Asset,now
from talkinghead_v2.server.importer import validate_import
from talkinghead_v2.server.lifecycle import Consent,ManagedAsset,Revision,Job,Upload,Lifecycle

ROOT=Path(__file__).resolve().parents[3]


def fixture_bundle():
    root=ROOT/'src/talkinghead_v2/fixtures/p03-avatar'
    return json.dumps({'schema_version':1,'manifest':(root/'avatar.json').read_text(),'mesh':(root/'mesh.json').read_text()}).encode()


async def intake(c,*,raw=None,seed='one'):
    raw=fixture_bundle() if raw is None else raw
    consent=await c.post('/v1/consents',json={'kind':'synthetic_fixture','reference':'first-party-mathematical-fixture',
        'allowed_uses':['timing_fixture_only'],'attested':True},headers={'Idempotency-Key':'consent-'+seed})
    assert consent.status_code==201,consent.text
    body={'consent_id':consent.json()['id'],'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    upload=await c.post('/v1/uploads',json=body,headers={'Idempotency-Key':'upload-'+seed})
    assert upload.status_code==201,upload.text
    data=upload.json()
    put=await c.put(data['content_url'],content=raw,headers={'Content-Type':'application/json','X-Upload-Token':data['write_token']})
    assert put.status_code==200,put.text
    assert (await c.post(f"/v1/uploads/{data['id']}/complete",json={})).status_code==200
    return consent.json(),data,body


async def import_ready(c,app,*,seed='one'):
    consent,upload,_=await intake(c,seed=seed)
    body={'upload_id':upload['id'],'name':'Fixture '+seed}
    response=await c.post('/v1/enrollments',json=body,headers={'Idempotency-Key':'enroll-'+seed})
    assert response.status_code==202,response.text
    job=response.json()
    claim=await asyncio.to_thread(app.state.lifecycle.claim)
    assert claim[0].id==job['id']
    revision=await asyncio.to_thread(validate_import,app.state.lifecycle,claim)
    return consent,upload,job,revision,body


async def test_real_import_review_session_revoke_and_durable_deletion(client):
    c,app=client;service=app.state.lifecycle
    consent,upload,job,revision,body=await import_ready(c,app)
    detail=await c.get('/v1/assets/'+job['asset_id'])
    assert detail.status_code==200 and detail.json()['evaluation']['structural_compatibility']
    assert not detail.json()['evaluation']['human_quality_qualified']
    preview=await c.get('/v1/assets/'+job['asset_id']+'/preview')
    assert preview.content.startswith(b'\x89PNG') and preview.headers['cache-control']=='no-store'
    retry=await c.post('/v1/enrollments',json=body,headers={'Idempotency-Key':'enroll-one'})
    assert retry.json()['id']==job['id']
    approval={'revision':revision,'accept_static_reference_limitations':True}
    for _ in range(2):
        response=await c.post('/v1/assets/'+job['asset_id']+'/approve',json=approval,headers={'Idempotency-Key':'review-one'})
        assert response.status_code==200,response.text
    first=await c.post('/v1/sessions',json={'asset_id':job['asset_id']},headers={'Idempotency-Key':'session-one'})
    assert first.status_code==202,first.text
    sid=first.json()['id'];await asyncio.sleep(.01)
    assert (await c.post('/v1/sessions/'+sid+'/join-grant',json={})).status_code==200
    revoked=await c.post('/v1/consents/'+consent['id']+'/revoke',json={})
    assert revoked.status_code==202 and job['asset_id'] in revoked.json()['affected_assets']
    app.state.actors[sid].media.fence.assert_called_with(permanent=True)
    assert (await c.post('/v1/sessions/'+sid+'/join-grant',json={})).status_code==409
    assert (await c.get('/v1/assets/'+job['asset_id']+'/preview')).status_code==403
    assert (await c.post('/v1/sessions',json={'asset_id':job['asset_id']},headers={'Idempotency-Key':'session-two'})).status_code==403
    assert (await c.post('/v1/sessions',json={},headers={'Idempotency-Key':'session-default'})).status_code==503
    event=next(e for e in service.outbox() if e.kind=='revoke_asset')
    with pytest.raises(RuntimeError,match='cleanup'): service.purge_and_receipt(event)
    await app.state.actors[sid].close()
    service.purge_and_receipt(event)
    receipt=await c.get('/v1/assets/'+job['asset_id']+'/deletion')
    assert receipt.json()['status']=='complete'
    assert not service.upload_path(event.tenant_id,upload['id']).exists()
    service.purge_consent(next(e for e in service.outbox() if e.kind=='revoke_consent'))
    restored=Lifecycle(app.state.store,service.root,service.tts_revision)
    with pytest.raises(PermissionError): restored.serving_manifest(event.tenant_id,job['asset_id'],revision)
    assert (await c.post('/v1/assets/'+job['asset_id']+'/approve',json=approval,headers={'Idempotency-Key':'review-one'})).status_code==403


async def test_cancel_and_revoke_fence_inflight_attempt_and_prevent_object_recreation(client):
    c,app=client;service=app.state.lifecycle
    _,upload,_=await intake(c)
    response=await c.post('/v1/enrollments',json={'upload_id':upload['id'],'name':'Cancelled'},headers={'Idempotency-Key':'enroll'})
    job=response.json();claim=service.claim()
    assert (await c.post('/v1/jobs/'+job['id']+'/cancel',json={})).json()['status']=='cancelled'
    path=service.root/claim[1].tenant_id/'assets'/claim[1].id/'stale'/'avatar.json'
    with pytest.raises(PermissionError): service.write_import_object(claim,path,b'{}')
    assert not path.exists()
    with pytest.raises(PermissionError): service.finish(claim,'a'*64,{})
    assert (await c.post('/v1/assets/'+job['asset_id']+'/revoke',json={})).status_code==202
    service.purge_and_receipt(service.outbox()[0])
    with pytest.raises(PermissionError): service.write_import_object(claim,path,b'{}')
    assert not path.parent.exists()


async def test_expired_attempt_is_fenced_and_retry_does_not_duplicate_publication(client):
    c,app=client;service=app.state.lifecycle
    _,upload,_=await intake(c)
    response=await c.post('/v1/enrollments',json={'upload_id':upload['id'],'name':'Retry'},headers={'Idempotency-Key':'enroll'})
    job=response.json();old=service.claim()
    with service.transaction() as db: db.get(Job,job['id']).lease_expires=now()-timedelta(seconds=1)
    fresh=service.claim()
    assert fresh[0].fence==old[0].fence+1 and fresh[0].attempt==2
    with pytest.raises(PermissionError): service.finish(old,'b'*64,{})
    revision=validate_import(service,fresh)
    assert [row['status'] for row in service.history(fresh[0].tenant_id,job['id'])]==['expired','complete']
    def approve(index):
        try: return service.approve(fresh[0].tenant_id,job['asset_id'],'same-review',{'revision':revision,'accept_static_reference_limitations':True}).status
        except Exception as exc: return type(exc).__name__
    with ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(approve,range(4)))
    assert results==['approved']*4
    with service.store.sessions() as db:
        assert len(list(db.scalars(select(Revision).where(Revision.asset_id==job['asset_id']))))==1
    assert (await c.post('/v1/enrollments',json={'upload_id':upload['id'],'name':'Other'},headers={'Idempotency-Key':'different'})).status_code==409


async def test_cross_tenant_denial_upload_grants_and_login_identity_reuse(client):
    c,app=client;service=app.state.lifecycle
    _,upload,body=await intake(c)
    old_token=upload['write_token']
    renewed=await c.post('/v1/uploads',json=body,headers={'Idempotency-Key':'upload-one'})
    assert renewed.json()['id']==upload['id'] and renewed.json()['write_token']!=old_token
    assert (await c.put(upload['content_url'],content=fixture_bundle(),headers={'Content-Type':'application/json','X-Upload-Token':old_token})).status_code==403
    token=renewed.json()['write_token']
    assert (await c.put(upload['content_url'],content=b'bad',headers={'Content-Type':'application/json','X-Upload-Token':token})).status_code==409
    first=await c.post('/v1/dev-login',json={});second=await c.post('/v1/dev-login',json={},headers={'X-CSRF-Token':first.json()['csrf_token']})
    assert first.json()['tenant_id']==second.json()['tenant_id'] and first.json()['csrf_token']==second.json()['csrf_token']
    c.cookies.clear();login=await c.post('/v1/dev-login',json={});c.headers['X-CSRF-Token']=login.json()['csrf_token']
    assert (await c.post('/v1/enrollments',json={'upload_id':upload['id'],'name':'Steal'},headers={'Idempotency-Key':'steal'})).status_code==404
    assert (await c.put(upload['content_url'],content=fixture_bundle(),headers={'Content-Type':'application/json','X-Upload-Token':token})).status_code==404


async def test_invalid_geometry_duplicate_keys_and_oversized_upload_fail_closed(client):
    c,app=client
    denied=await c.post('/v1/consents',json={'kind':'synthetic_fixture','reference':'   ',
        'allowed_uses':['timing_fixture_only'],'attested':True},headers={'Idempotency-Key':'blank'})
    assert denied.status_code==422
    bundle=json.loads(fixture_bundle());mesh=json.loads(bundle['mesh']);mesh['faces'][0]=[0,1,999999]
    bundle['mesh']=json.dumps(mesh)
    manifest=json.loads(bundle['manifest']);manifest['appearance']['sha256']='sha256:'+hashlib.sha256(bundle['mesh'].encode()).hexdigest()
    bundle['manifest']=json.dumps(manifest)
    _,upload,_=await intake(c,raw=json.dumps(bundle).encode())
    job=(await c.post('/v1/enrollments',json={'upload_id':upload['id'],'name':'Invalid'},headers={'Idempotency-Key':'invalid'})).json()
    claim=app.state.lifecycle.claim()
    with pytest.raises(ValueError,match='face index'): validate_import(app.state.lifecycle,claim)
    app.state.lifecycle.fail(claim,'invalid_import')
    assert (await c.get('/v1/jobs/'+job['id'])).json()['status']=='failed'
    assert (await c.put(upload['content_url'],content=b'x'*(4*1024*1024+1),headers={'Content-Type':'application/json','X-Upload-Token':upload['write_token']})).status_code==413
    from talkinghead_v2.server.lifecycle import strict_json
    with pytest.raises(ValueError,match='duplicate'): strict_json('{"key":1,"key":2}')
    with pytest.raises(ValueError,match='nonfinite'): strict_json('{"key":NaN}')


async def test_revoke_wins_concurrent_approval_and_pending_capture_is_purged(client):
    c,app=client;service=app.state.lifecycle
    _,_,job,revision,_=await import_ready(c,app)
    tenant=service.claim()  # No queued jobs remain.
    assert tenant is None
    asset=(await c.get('/v1/assets/'+job['asset_id'])).json()
    owner=service.get(ManagedAsset,app.state.store.authenticate(c.cookies.get('tha_dev')).id,job['asset_id']).tenant_id
    barrier=Barrier(2)
    def approve():
        barrier.wait()
        try: service.approve(owner,job['asset_id'],'racing-review',{'revision':revision,'accept_static_reference_limitations':True})
        except PermissionError: pass
    def revoke(): barrier.wait();service.revoke(owner,job['asset_id'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(approve);b=pool.submit(revoke);a.result();b.result()
    assert service.get(ManagedAsset,owner,job['asset_id']).status=='revoked'
    with service.store.sessions() as db:
        serving=db.get(Asset,(owner,job['asset_id']))
        assert serving is None or serving.status=='revoked'
    consent,upload,_=await intake(c,seed='unused-capture')
    assert service.upload_path(owner,upload['id']).is_file()
    await c.post('/v1/consents/'+consent['id']+'/revoke',json={})
    event=next(e for e in service.outbox() if e.kind=='revoke_consent' and e.resource_id==consent['id'])
    service.purge_consent(event)
    assert not service.upload_path(owner,upload['id']).exists()


async def test_revocation_during_track_publication_cannot_start_video_or_leave_audio(settings):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from talkinghead_v2.server.transport import Media
    from talkinghead_v2.contracts import GenerationKey
    media=Media(settings,'race-room','race-session')
    entered=asyncio.Event();release=asyncio.Event()
    async def publish(*args):
        entered.set();await release.wait();return SimpleNamespace(sid='audio-race')
    publish_mock=AsyncMock(side_effect=publish);unpublish=AsyncMock()
    media.room=SimpleNamespace(local_participant=SimpleNamespace(publish_track=publish_mock,unpublish_track=unpublish))
    pending=asyncio.create_task(media.start(GenerationKey('race-session','turn',1)))
    await entered.wait();media.fence(permanent=True);release.set()
    with pytest.raises(PermissionError): await pending
    assert media.blocked and media.audio is None and media.video is None
    publish_mock.assert_awaited_once();unpublish.assert_awaited_once_with('audio-race')
    with pytest.raises(PermissionError): await media.start(GenerationKey('race-session','new-turn',2))


def test_atomic_object_publication_recovers_from_interrupted_link(settings,tmp_path,monkeypatch):
    from talkinghead_v2.server.db import Store
    import os
    store=Store(settings.database_url);service=Lifecycle(store,tmp_path/'objects','0'*64)
    destination=service.root/'tenant'/'data'
    original=os.link
    def interrupted(*args,**kwargs): raise OSError('injected pre-publication crash')
    monkeypatch.setattr(os,'link',interrupted)
    with pytest.raises(OSError): service.write_object(destination,b'complete immutable bytes')
    assert not destination.exists() and not list(destination.parent.glob('.pending-*'))
    monkeypatch.setattr(os,'link',original)
    service.write_object(destination,b'complete immutable bytes')
    service.write_object(destination,b'complete immutable bytes')
    assert destination.read_bytes()==b'complete immutable bytes'
    with pytest.raises(ValueError): service.write_object(destination,b'different bytes')
    store.engine.dispose()


async def test_retry_budget_and_expired_unenrolled_upload_cleanup(client):
    c,app=client;service=app.state.lifecycle
    _,upload,_=await intake(c)
    job=(await c.post('/v1/enrollments',json={'upload_id':upload['id'],'name':'Lease failures'},headers={'Idempotency-Key':'leases'})).json()
    for expected in range(1,4):
        claim=service.claim();assert claim[0].attempt==expected
        with service.transaction() as db: db.get(Job,job['id']).lease_expires=now()-timedelta(seconds=1)
    assert service.claim() is None
    assert (await c.get('/v1/jobs/'+job['id'])).json()['error_code']=='attempts_exhausted'
    assert (await c.get('/v1/assets/'+job['asset_id'])).json()['status']=='failed'
    _,unused,_=await intake(c,seed='expired')
    owner=app.state.store.authenticate(c.cookies.get('tha_dev')).id
    with service.transaction() as db: db.get(Upload,unused['id']).expires=now()-timedelta(seconds=1)
    service.expire_uploads()
    assert not service.upload_path(owner,unused['id']).exists()


@pytest.mark.parametrize('tag',['failure','error','skipped'])
def test_evidence_export_refuses_nonpassing_junit(tmp_path,tag):
    import importlib.util
    spec=importlib.util.spec_from_file_location('p05_qualification',ROOT/'scripts/qualify_p05.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=tmp_path/'result.xml';path.write_text(f'<testsuite><testcase><{tag} message="fixture" /></testcase></testsuite>')
    with pytest.raises(ValueError,match='actually pass'): module.tests(path)
