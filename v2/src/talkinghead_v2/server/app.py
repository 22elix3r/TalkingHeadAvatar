"""Loopback-only FastAPI control plane with PostgreSQL and bounded live actors."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import hmac
import json
import secrets
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI,Request,HTTPException
from fastapi.responses import JSONResponse,StreamingResponse,RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,ConfigDict,Field
from livekit import api

from talkinghead_v2.providers.tts.native import load_native_config
from talkinghead_v2.assets.artifacts import sha256_file
from talkinghead_v2.server.actor import Actor
from talkinghead_v2.server.config import Settings
from talkinghead_v2.server.db import Store,hashed
from talkinghead_v2.server.providers import Asr,Llm
from talkinghead_v2.server.transport import grant


class NewSession(BaseModel):
    model_config=ConfigDict(extra='forbid')
    asset_id: str='synthetic-audio-reference'
    profile: str='audio_reference'


class Command(BaseModel):
    model_config=ConfigDict(extra='forbid')
    sequence: int=Field(ge=1,le=10000)
    expected_epoch: int=Field(ge=0)
    kind: str
    text: str|None=Field(default=None,max_length=1500)


def create_app(settings: Settings,*,load_models=True):
    store=Store(settings.database_url)
    actors={}
    resources={'ready':False}
    @asynccontextmanager
    async def lifespan(app):
        if load_models:
            config=await asyncio.to_thread(load_native_config,settings.tts_runtime)
            if config.voice_kind!='synthetic_fixture': raise ValueError('P04 audio_reference requires an explicitly labeled synthetic voice fixture')
            asr=await asyncio.to_thread(Asr,settings.asr_path,settings.asr_files)
            llm=Llm(settings);await llm.prepare()
            resources.update(config=config,asr=asr,llm=llm,asset_revision=sha256_file(settings.tts_runtime))
        lk=api.LiveKitAPI(settings.livekit_url.replace('ws://','http://'),settings.livekit_key,settings.livekit_secret)
        resources['lk']=lk
        for room,sid in await asyncio.to_thread(store.recover):
            try: await lk.room.delete_room(api.DeleteRoomRequest(room=room))
            except Exception: pass
        resources['ready']=load_models
        async def reap():
            while True:
                await asyncio.sleep(1)
                for sid,actor in list(actors.items()):
                    if actor.closed and getattr(actor,'close_done',None) and actor.close_done.is_set(): actors.pop(sid,None)
        reaper=asyncio.create_task(reap())
        yield
        resources['ready']=False
        reaper.cancel();await asyncio.gather(reaper,return_exceptions=True)
        await asyncio.gather(*(a.close() for a in list(actors.values())),return_exceptions=True)
        if load_models:
            asr.close();await llm.close()
        await lk.aclose();store.engine.dispose()
    app=FastAPI(title='TalkingHeadAvatar local research reference',lifespan=lifespan,docs_url=None,redoc_url=None)
    app.state.store=store;app.state.actors=actors;app.state.resources=resources

    @app.middleware('http')
    async def local_only(request: Request,call_next):
        host=request.headers.get('host','').split(':')[0]
        if request.client.host not in {'127.0.0.1','::1'} or host not in {'127.0.0.1','localhost'}:
            return JSONResponse({'detail':'loopback_development_only'},status_code=403)
        origin=request.headers.get('origin')
        if origin and origin!=settings.origin:
            return JSONResponse({'detail':'origin_denied'},status_code=403)
        # A navigation to localhost should enter the configured cookie/CSRF origin,
        # rather than loading a UI whose first mutation will then be rejected.
        if request.method=='GET' and request.url.path=='/' and str(request.url).rstrip('/')!=settings.origin.rstrip('/'):
            return RedirectResponse(settings.origin+'/',status_code=307)
        length=request.headers.get('content-length','0')
        if not length.isdigit() or int(length)>8192:
            return JSONResponse({'detail':'payload_too_large'},status_code=413)
        if request.method=='POST':
            body=bytearray()
            async for chunk in request.stream():
                if len(body)+len(chunk)>8192:
                    return JSONResponse({'detail':'payload_too_large'},status_code=413)
                body.extend(chunk)
            request._body=bytes(body)
        response=await call_next(request)
        response.headers['Cache-Control']='no-store' if request.url.path.startswith('/v1') else 'no-cache'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        return response

    async def principal(request,*,mutation=False):
        cookie=request.cookies.get('tha_dev','')
        p=await asyncio.to_thread(store.authenticate,cookie)
        if not p: raise HTTPException(401,'login_required')
        if mutation:
            if request.headers.get('origin')!=settings.origin: raise HTTPException(403,'origin_required')
            csrf=request.headers.get('x-csrf-token','')
            if not hmac.compare_digest(hashed(csrf),p.csrf_hash): raise HTTPException(403,'csrf_denied')
        return p
    async def owned(request,sid,*,mutation=False):
        p=await principal(request,mutation=mutation)
        row=await asyncio.to_thread(store.get,sid,p.id)
        if not row: raise HTTPException(404,'session_not_found')
        return row
    def summary(row):
        return {'id':row.id,'status':row.status,'epoch':row.epoch,'profile':'audio_reference',
            'asset_revision':row.asset_revision,'error_code':row.error_code,
            'visual_quality_qualified':False,'conversation_slo_qualified':False}

    @app.get('/health/live')
    async def live(): return {'status':'live'}
    @app.get('/health/ready')
    async def ready():
        return JSONResponse({'ready':resources['ready'],'profile':'audio_reference','avatar_ready':False},status_code=200 if resources['ready'] else 503)
    @app.get('/v1/capabilities')
    async def capabilities():
        return {'profiles':['audio_reference'],'voice_kind':'synthetic_fixture','avatar_ready':False,
                'authentication':'loopback_development','recording':False,'asr_streaming':'rolling_window_revisions',
                'llm':'explicit_cpu_qwen','tts':'Qwen3-TTS Base 1.7B','receiver_resume':'replace_tracks'}
    @app.post('/v1/dev-login')
    async def login(request: Request):
        if request.headers.get('origin')!=settings.origin: raise HTTPException(403,'origin_required')
        revision=resources.get('asset_revision','0'*64)
        _,cookie,csrf=await asyncio.to_thread(store.login,revision)
        response=JSONResponse({'csrf_token':csrf,'mode':'loopback_development'})
        response.set_cookie('tha_dev',cookie,httponly=True,samesite='strict',max_age=28800)
        return response
    @app.get('/v1/assets')
    async def assets(request: Request):
        await principal(request)
        return {'items':[{'id':'synthetic-audio-reference','name':'Synthetic demo voice','status':'approved',
                           'visual_mode':'static_procedural_reference','allowed_uses':['synthetic_plumbing_only']}]}
    @app.post('/v1/sessions')
    async def new_session(body: NewSession,request: Request):
        p=await principal(request,mutation=True)
        if body.profile!='audio_reference': raise HTTPException(422,'profile_unavailable')
        if not resources['ready']: raise HTTPException(503,'providers_not_ready')
        key=request.headers.get('idempotency-key','')
        if not 1<=len(key)<=100: raise HTTPException(422,'idempotency_key_required')
        try:
            row,created=await asyncio.to_thread(store.reserve,p.id,key,hashed(body.model_dump_json()),body.asset_id)
        except ValueError: raise HTTPException(409,'idempotency_conflict')
        except PermissionError: raise HTTPException(403,'asset_unavailable')
        except RuntimeError: raise HTTPException(503,'capacity_exhausted')
        if created:
            actor=Actor(settings,store,row,resources['asr'],resources['llm'],resources['config'])
            actors[row.id]=actor
            actor.preparation=asyncio.create_task(actor.prepare())
        return JSONResponse(summary(row),status_code=202 if row.status=='PREPARING' else 200)
    @app.get('/v1/sessions/{sid}')
    async def status(sid: str,request: Request): return summary(await owned(request,sid))
    @app.post('/v1/sessions/{sid}/join-grant')
    async def join(sid: str,request: Request):
        row=await owned(request,sid,mutation=True)
        if row.status not in {'READY','ACTIVE'}: raise HTTPException(409,'session_not_ready')
        actor=actors.get(sid)
        if not actor or actor.closed or getattr(actor,'closing',False): raise HTTPException(409,'session_closed')
        return {'url':settings.livekit_url,'token':grant(settings,row.room,'user-'+sid,'browser'),
                'publisher':'media-'+sid,'epoch':actor.epoch}
    @app.post('/v1/sessions/{sid}/heartbeat')
    async def heartbeat(sid: str,request: Request):
        import time
        await owned(request,sid,mutation=True)
        actor=actors.get(sid)
        if not actor or actor.closed: raise HTTPException(409,'session_closed')
        actor.last_browser_heartbeat=time.monotonic()
        return {'epoch':actor.epoch}
    @app.get('/v1/sessions/{sid}/events')
    async def events(sid: str,request: Request):
        await owned(request,sid)
        actor=actors.get(sid)
        if not actor: raise HTTPException(409,'session_unavailable')
        last=request.headers.get('last-event-id','0')
        if not last.isdigit(): raise HTTPException(422,'invalid_event_cursor')
        return StreamingResponse(actor.events.stream(int(last)),media_type='text/event-stream',headers={'X-Accel-Buffering':'no'})
    @app.post('/v1/sessions/{sid}/controls')
    async def controls(sid: str,body: Command,request: Request):
        row=await owned(request,sid,mutation=True)
        actor=actors.get(sid)
        if not actor or actor.closed: raise HTTPException(409,'session_closed')
        async with actor.control_lock:
            digest=hashed(body.model_dump_json())
            previous=await asyncio.to_thread(store.control,sid,body.sequence)
            if previous:
                if previous.body_hash!=digest: raise HTTPException(409,'control_sequence_conflict')
                return previous.result
            if body.sequence<=await asyncio.to_thread(store.last_control_sequence,sid): raise HTTPException(409,'stale_control_sequence')
            if body.expected_epoch!=actor.epoch: raise HTTPException(409,'epoch_conflict')
            if body.kind=='interrupt': await actor.interrupt()
            elif body.kind=='text' and body.text and body.text.strip(): await actor.answer(body.text.strip())
            else: raise HTTPException(422,'invalid_control')
            result={'epoch':actor.epoch,'sequence':body.sequence,'receiver_safe_resume':False}
            await asyncio.to_thread(store.record_control,sid,body.sequence,digest,result)
            return result
    @app.post('/v1/sessions/{sid}/close')
    async def close(sid: str,request: Request):
        await owned(request,sid,mutation=True)
        actor=actors.get(sid)
        if actor and await actor.close() is False: raise HTTPException(503,'close_persistence_pending')
        return {'status':'CLOSED'}
    if settings.web_dist.is_dir(): app.mount('/',StaticFiles(directory=settings.web_dist,html=True),name='web')
    return app
