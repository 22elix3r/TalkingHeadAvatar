import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import httpx
import jwt
import pytest
from sqlalchemy import text

from talkinghead_v2.server.config import Settings
from talkinghead_v2.server.db import Store,hashed
from talkinghead_v2.server.transport import CheckedReceiveQueue,AudioDiscontinuity
from talkinghead_v2.server.actor import Events
from talkinghead_v2.server import app as module


ROOT=Path(__file__).resolve().parents[3]


@pytest.fixture
def settings():
    cfg=Settings.load(ROOT/'var/p04/test-runtime.json')
    assert cfg.database_url.endswith('/tha_p04_test'), 'never reset the running reference database'
    store=Store(cfg.database_url)
    with store.engine.begin() as db:
        db.execute(text('TRUNCATE principals,asset_revisions,sessions,idempotency_records,session_controls'))
    store.engine.dispose()
    return cfg


@pytest.fixture
async def client(settings,monkeypatch):
    class FakeActor:
        def __init__(self,config,store,row,*providers):
            self.store,self.row=store,row;self.epoch=0;self.closed=False;self.control_lock=asyncio.Lock();self.events=Events()
        async def prepare(self): self.store.update(self.row.id,status='READY')
        async def interrupt(self):
            self.epoch+=1;self.store.update(self.row.id,epoch=self.epoch)
        async def answer(self,text): await self.interrupt()
        async def close(self): self.closed=True;self.store.update(self.row.id,status='CLOSED')
    monkeypatch.setattr(module,'Actor',FakeActor)
    app=module.create_app(settings,load_models=False)
    async with app.router.lifespan_context(app):
        app.state.resources.update(ready=True,asr=None,llm=None,config=None)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app,client=('127.0.0.1',123)),base_url=settings.origin) as c:
            c.headers['Origin']=settings.origin
            login=await c.post('/v1/dev-login',json={})
            c.headers['X-CSRF-Token']=login.json()['csrf_token']
            yield c,app


async def test_real_postgres_idempotency_cross_tenant_and_grant_scopes(client,settings):
    c,app=client
    headers={'Idempotency-Key':'same-key'}
    first=await c.post('/v1/sessions',json={},headers=headers)
    assert first.status_code==202
    sid=first.json()['id'];await asyncio.sleep(.01)
    same=await c.post('/v1/sessions',json={},headers=headers)
    assert same.json()['id']==sid and len(app.state.actors)==1
    different=await c.post('/v1/sessions',json={'asset_id':'unknown'},headers=headers)
    assert different.status_code==409
    grant=await c.post(f'/v1/sessions/{sid}/join-grant',json={})
    token=jwt.decode(grant.json()['token'],settings.livekit_secret,algorithms=['HS256'],audience=None,options={'verify_aud':False})
    assert token['video']['canPublishSources']==['microphone'] and token['video']['canPublishData'] is False
    assert token['sub']=='user-'+sid
    previous=await c.post(f'/v1/sessions/{sid}/controls',json={'sequence':1,'expected_epoch':0,'kind':'interrupt'})
    repeated=await c.post(f'/v1/sessions/{sid}/controls',json={'sequence':1,'expected_epoch':0,'kind':'interrupt'})
    assert previous.json()==repeated.json() and previous.json()['epoch']==1
    conflict=await c.post(f'/v1/sessions/{sid}/controls',json={'sequence':1,'expected_epoch':0,'kind':'text','text':'different'})
    assert conflict.status_code==409
    high=await c.post(f'/v1/sessions/{sid}/controls',json={'sequence':3,'expected_epoch':1,'kind':'interrupt'})
    assert high.json()['epoch']==2
    reordered=await c.post(f'/v1/sessions/{sid}/controls',json={'sequence':2,'expected_epoch':2,'kind':'interrupt'})
    assert reordered.status_code==409
    await c.post('/v1/dev-login',json={})  # A different cookie means a different tenant.
    assert (await c.get(f'/v1/sessions/{sid}')).status_code==404


async def test_loopback_origin_csrf_body_and_unsupported_profile_fail_closed(client):
    c,app=client
    assert (await c.get('/v1/assets',headers={'Host':'attacker.example'})).status_code==403
    assert (await c.post('/v1/sessions',json={},headers={'Origin':'https://example.org'})).status_code==403
    assert (await c.post('/v1/sessions',json={},headers={'X-CSRF-Token':'wrong'})).status_code==403
    assert (await c.post('/v1/sessions',content=b'x'*9000)).status_code==413
    async def chunks():
        for _ in range(9): yield b'x'*1024
    assert (await c.post('/v1/sessions',content=chunks())).status_code==413
    assert (await c.post('/v1/sessions',json={'profile':'photorealistic'})).status_code==422
    assert not app.state.actors


def test_atomic_single_slot_admission_and_stale_fence(settings):
    store=Store(settings.database_url)
    tenant,*_=store.login('a'*40)
    def reserve(index):
        try: return store.reserve(tenant,str(index),'b'*64,'synthetic-audio-reference')[0]
        except RuntimeError: return None
    with ThreadPoolExecutor(max_workers=8) as pool: results=list(pool.map(reserve,range(8)))
    rows=[r for r in results if r is not None]
    assert len(rows)==1
    row=rows[0]
    with pytest.raises(RuntimeError,match='stale_session_fence'): store.update(row.id,fence=0)
    recovered=store.recover()
    assert recovered==[(row.room,row.id)]
    with pytest.raises(RuntimeError,match='stale_session_fence'): store.update(row.id,fence=1)
    store.engine.dispose()


async def test_receive_queue_overflow_signals_discontinuity_instead_of_silent_eviction():
    queue=CheckedReceiveQueue(max_frames=2)
    queue.put('first');queue.put('second');queue.put('overflow')
    assert queue.failed and queue.lost_frames==3 and queue.peak==2
    with pytest.raises(AudioDiscontinuity): await queue.get()
    assert not queue.queue


def test_public_bind_configuration_is_rejected(settings,tmp_path):
    import json
    source=json.loads((ROOT/'var/p04/test-runtime.json').read_text())
    source['origin']='http://0.0.0.0:18700'
    path=tmp_path/'config.json';path.write_text(json.dumps(source))
    with pytest.raises(ValueError,match='loopback'): Settings.load(path)


async def test_localhost_navigation_uses_canonical_origin_but_cannot_bypass_csrf(client,settings):
    c,_=client
    c.headers.pop('Origin')
    response=await c.get('http://localhost:18700/',follow_redirects=False)
    assert response.status_code==307 and response.headers['location']==settings.origin+'/'
    denied=await c.post('/v1/dev-login',json={},headers={'Origin':'http://localhost:18700'})
    assert denied.status_code==403 and denied.json()['detail']=='origin_denied'


async def test_concurrent_close_stops_sources_once_even_when_database_fails():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock,Mock
    from talkinghead_v2.server.actor import Actor
    actor=object.__new__(Actor)
    actor.closing=False;actor.closed=False;actor.closed_persisted=False;actor.close_done=asyncio.Event()
    actor.preparation=actor.input_task=actor.partial=actor.final_task=actor.heartbeat=None
    actor.epoch=0;actor.row=SimpleNamespace(id='fault-fixture')
    actor.tts=SimpleNamespace(close=AsyncMock());actor.media=SimpleNamespace(close=AsyncMock())
    actor.controller=SimpleNamespace(disconnect=AsyncMock());actor.events=Events()
    actor.store=SimpleNamespace(update=Mock(side_effect=RuntimeError('database unavailable')))
    async def interrupt(*args): await asyncio.sleep(.01)
    actor.interrupt=AsyncMock(side_effect=interrupt)
    outcomes=await asyncio.gather(actor.close(),actor.close())
    assert outcomes==[False,False] and actor.closed and actor.close_done.is_set()
    actor.tts.close.assert_awaited_once();actor.media.close.assert_awaited_once()
    actor.controller.disconnect.assert_awaited_once();actor.interrupt.assert_awaited_once()
