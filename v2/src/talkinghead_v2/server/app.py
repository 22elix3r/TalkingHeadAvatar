"""Loopback-only FastAPI control plane with PostgreSQL and bounded live actors."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import hmac
import json
import re
import secrets
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI,Request,HTTPException,Query
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
from talkinghead_v2.server.lifecycle import Lifecycle,ManagedAsset,MAX_UPLOAD
from talkinghead_v2.server.lifecycle_api import routes as lifecycle_routes,worker as lifecycle_worker,asset_view
from talkinghead_v2.assets.avatar import load_avatar
from talkinghead_v2.rendering.mesh import MeshRenderer
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


def create_app(settings: Settings,*,load_models=True,run_jobs=True):
    research=settings.media_backend=='native_claire_research'
    profile='claire_vendor_research' if research else 'audio_reference'
    reference_id='claire-vendor-research' if research else 'synthetic-audio-reference'
    cookie_name='tha_dev_claire' if research else 'tha_dev'
    csrf_cookie='tha_csrf_claire' if research else 'tha_csrf'
    store=Store(settings.database_url)
    actors={}
    resources={'ready':False}
    mutation_lock=asyncio.Lock()
    lifecycle=Lifecycle(store,settings.private_dir/'asset-store','0'*64)
    @asynccontextmanager
    async def lifespan(app):
        await asyncio.to_thread(store.require_schema)
        if load_models:
            config=await asyncio.to_thread(load_native_config,settings.tts_runtime)
            if config.voice_kind!='synthetic_fixture': raise ValueError('P04 audio_reference requires an explicitly labeled synthetic voice fixture')
            asr=await asyncio.to_thread(Asr,settings.asr_path,settings.asr_files)
            llm=Llm(settings);await llm.prepare()
            resources.update(config=config,asr=asr,llm=llm,asset_revision=sha256_file(settings.tts_runtime))
            resources['tts_revision']=resources['asset_revision']
            if research:
                from talkinghead_v2.server.claire_media import ClaireMediaConfig
                avatar=await asyncio.to_thread(ClaireMediaConfig.load,settings.native_media_runtime)
                await asyncio.to_thread(avatar.validate_device)
                resources.update(avatar_config=avatar,asset_revision=hashed(resources['tts_revision']+':'+avatar.revision))
        lifecycle.tts_revision=resources.get('tts_revision',resources.get('asset_revision','0'*64))
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
        jobs=asyncio.create_task(lifecycle_worker(lifecycle,actors,lk)) if run_jobs and not research else None
        yield
        resources['ready']=False
        reaper.cancel();await asyncio.gather(reaper,return_exceptions=True)
        if jobs:
            jobs.cancel();await asyncio.gather(jobs,return_exceptions=True)
        await asyncio.gather(*(a.close() for a in list(actors.values())),return_exceptions=True)
        if load_models:
            asr.close();await llm.close()
        await lk.aclose();store.engine.dispose()
    app=FastAPI(title='TalkingHeadAvatar local research reference',lifespan=lifespan,docs_url=None,redoc_url=None)
    app.state.store=store;app.state.actors=actors;app.state.resources=resources
    app.state.lifecycle=lifecycle

    @app.middleware('http')
    async def local_only(request: Request,call_next):
        host=request.headers.get('host','').split(':')[0]
        if request.client.host not in {'127.0.0.1','::1'} or host not in {'127.0.0.1','localhost'}:
            return JSONResponse({'detail':'loopback_development_only'},status_code=403)
        origin=request.headers.get('origin')
        if origin and origin!=settings.origin:
            return JSONResponse({'detail':'origin_denied'},status_code=403)
        if research and request.method in {'POST','PUT'} and request.url.path.startswith(('/v1/consents','/v1/uploads','/v1/enrollments','/v1/jobs','/v1/assets')):
            return JSONResponse({'detail':'research_reference_only'},status_code=422)
        # A navigation to localhost should enter the configured cookie/CSRF origin,
        # rather than loading a UI whose first mutation will then be rejected.
        if request.method=='GET' and request.url.path=='/' and str(request.url).rstrip('/')!=settings.origin.rstrip('/'):
            return RedirectResponse(settings.origin+'/',status_code=307)
        upload=request.method=='PUT' and re.fullmatch(r'/v1/uploads/[a-f0-9]{32}/content',request.url.path)
        limit=MAX_UPLOAD if upload else 8192
        length=request.headers.get('content-length','0')
        if not length.isdigit() or int(length)>limit:
            return JSONResponse({'detail':'payload_too_large'},status_code=413)
        if request.method in {'POST','PUT'}:
            body=bytearray()
            async for chunk in request.stream():
                if len(body)+len(chunk)>limit:
                    return JSONResponse({'detail':'payload_too_large'},status_code=413)
                body.extend(chunk)
            request._body=bytes(body)
        response=await call_next(request)
        response.headers['Cache-Control']='no-store' if request.url.path.startswith('/v1') else 'no-cache'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        return response

    async def principal(request,*,mutation=False):
        cookie=request.cookies.get(cookie_name,'')
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
        return {'id':row.id,'status':row.status,'epoch':row.epoch,'profile':profile,
            'asset_revision':row.asset_revision,'error_code':row.error_code,
            'visual_quality_qualified':False,'conversation_slo_qualified':False}

    @app.get('/health/live')
    async def live(): return {'status':'live'}
    @app.get('/health/ready')
    async def ready():
        return JSONResponse({'ready':resources['ready'],'profile':profile,'avatar_ready':research and resources['ready'],
                             'visual_quality_qualified':False},status_code=200 if resources['ready'] else 503)
    @app.get('/v1/capabilities')
    async def capabilities():
        return {'profiles':[profile],'voice_kind':'synthetic_fixture','avatar_ready':research,
                'visual_quality_qualified':False,'asset_imports':not research,
                'authentication':'loopback_development','recording':False,'asr_streaming':'rolling_window_revisions',
                'llm':'explicit_cpu_qwen','tts':'Qwen3-TTS Base 1.7B','receiver_resume':'replace_tracks'}
    @app.post('/v1/dev-login')
    async def login(request: Request):
        if request.headers.get('origin')!=settings.origin: raise HTTPException(403,'origin_required')
        revision=resources.get('tts_revision',resources.get('asset_revision','0'*64))
        tenant,cookie,csrf=await asyncio.to_thread(store.resume_login,revision,request.cookies.get(cookie_name,''),request.headers.get('x-csrf-token',''))
        if research:
            try: await asyncio.to_thread(store.reference_asset,tenant,reference_id,resources.get('asset_revision','0'*64))
            except RuntimeError: raise HTTPException(409,'reference_active_revision_conflict')
        response=JSONResponse({'csrf_token':csrf,'tenant_id':tenant,'mode':'loopback_development',
                               'profile':profile,'default_asset_id':reference_id})
        response.set_cookie(cookie_name,cookie,httponly=True,samesite='strict',max_age=28800)
        response.set_cookie(csrf_cookie,csrf,httponly=False,samesite='strict',max_age=28800)
        return response
    @app.get('/v1/assets')
    async def assets(request: Request,after: str=Query('',pattern='^([a-f0-9]{32})?$'),limit: int=Query(50,ge=1,le=50)):
        p=await principal(request)
        if research:
            asset=await asyncio.to_thread(store.reference_asset,p.id,reference_id)
            avatar=resources.get('avatar_config')
            return {'items':[] if asset is None else [{'id':reference_id,'name':'Claire research avatar · synthetic voice',
                'status':asset.status,'revision':asset.revision,'visual_mode':'claire_vendor_research','builtin':True,
                'supports_imports':False,'allowed_uses':['local_noncommercial_research'],'consent_evidence':'vendor_assertion',
                'human_enrollment':False,'basis_hash':getattr(avatar,'basis_hash',None),'topology_hash':getattr(avatar,'topology_hash',None),
                'model_hash':getattr(avatar,'model_hash',None)}],'next_cursor':None}
        managed=await asyncio.to_thread(lifecycle.rows,ManagedAsset,p.id,after,limit)
        return {'items':[{'id':'synthetic-audio-reference','name':'Synthetic demo voice','status':'approved',
                           'visual_mode':'static_procedural_reference','allowed_uses':['synthetic_plumbing_only']},
                         *[asset_view(row) for row in managed]],'next_cursor':managed[-1].id if len(managed)==limit else None}
    @app.post('/v1/sessions')
    async def new_session(body: NewSession,request: Request):
        p=await principal(request,mutation=True)
        if body.profile!=profile: raise HTTPException(422,'profile_unavailable')
        if research and body.asset_id!=reference_id: raise HTTPException(422,'research_reference_only')
        if not resources['ready']: raise HTTPException(503,'providers_not_ready')
        key=request.headers.get('idempotency-key','')
        if not 1<=len(key)<=100: raise HTTPException(422,'idempotency_key_required')
        try:
            row,created=await asyncio.to_thread(store.reserve,p.id,key,hashed(body.model_dump_json()),body.asset_id)
        except ValueError: raise HTTPException(409,'idempotency_conflict')
        except PermissionError: raise HTTPException(403,'asset_unavailable')
        except RuntimeError: raise HTTPException(503,'capacity_exhausted')
        if created:
            async with mutation_lock:
                try:
                    selected=None
                    if body.asset_id not in {'synthetic-audio-reference',reference_id}:
                        path=await asyncio.to_thread(lifecycle.serving_manifest,p.id,row.asset_id,row.asset_revision)
                        selected=await asyncio.to_thread(load_avatar,path,tenant_id=p.id)
                    actor=Actor(settings,store,row,resources['asr'],resources['llm'],resources['config'],resources.get('avatar_config'))
                    if selected:
                        actor.media.asset=selected;actor.media.renderer=MeshRenderer(selected,resolution=128)
                    actors[row.id]=actor
                    actor.preparation=asyncio.create_task(actor.prepare())
                except (PermissionError,LookupError,ValueError):
                    await asyncio.to_thread(store.update,row.id,status='FAILED',error='asset_unavailable')
                    raise HTTPException(403,'asset_unavailable')
        return JSONResponse(summary(row),status_code=202 if row.status=='PREPARING' else 200)
    @app.get('/v1/sessions/{sid}')
    async def status(sid: str,request: Request): return summary(await owned(request,sid))
    @app.post('/v1/sessions/{sid}/join-grant')
    async def join(sid: str,request: Request):
        row=await owned(request,sid,mutation=True)
        if row.status not in {'READY','ACTIVE'}: raise HTTPException(409,'session_not_ready')
        if research:
            asset=await asyncio.to_thread(store.reference_asset,row.tenant_id,reference_id)
            if asset is None or asset.status!='approved' or asset.revision!=row.asset_revision: raise HTTPException(403,'asset_unavailable')
        elif row.asset_id!='synthetic-audio-reference':
            try: await asyncio.to_thread(lifecycle.serving_manifest,row.tenant_id,row.asset_id,row.asset_revision)
            except (PermissionError,LookupError): raise HTTPException(403,'asset_unavailable')
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
        if row.status not in {'READY','ACTIVE'}: raise HTTPException(409,'session_closed')
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
    app.include_router(lifecycle_routes(lifecycle,principal,actors,mutation_lock))
    if settings.web_dist.is_dir(): app.mount('/',StaticFiles(directory=settings.web_dist,html=True),name='web')
    return app
