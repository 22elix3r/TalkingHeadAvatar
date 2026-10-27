"""Authenticated bounded import endpoints and local durable worker coordination."""
import asyncio
import logging
from typing import Literal
from pathlib import Path

from fastapi import APIRouter,Request,HTTPException,Query
from fastapi.responses import Response
from livekit import api
from pydantic import BaseModel,ConfigDict,Field,model_validator

from talkinghead_v2.assets.avatar import load_avatar
from talkinghead_v2.contracts import ContractError
from talkinghead_v2.server.importer import validate_import
from talkinghead_v2.server.lifecycle import MAX_UPLOAD,Consent,Upload,ManagedAsset,Revision,Job,Outbox


class Input(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)


class ConsentInput(Input):
    kind: Literal['synthetic_fixture','human_consented','research_authorized']
    reference: str=Field(min_length=1,max_length=300)
    allowed_uses: list[Literal['timing_fixture_only','research_avatar']]=Field(min_length=1,max_length=1)
    attested: Literal[True]
    @model_validator(mode='after')
    def scope(self):
        expected='timing_fixture_only' if self.kind=='synthetic_fixture' else 'research_avatar'
        if self.allowed_uses!=[expected]: raise ValueError('consent scope must match provenance')
        return self


class UploadInput(Input):
    purpose: Literal['avatar_import']='avatar_import'
    content_type: Literal['application/json']='application/json'
    consent_id: str=Field(pattern='^[a-f0-9]{32}$')
    size: int=Field(strict=True,ge=1,le=MAX_UPLOAD)
    sha256: str=Field(pattern='^[a-f0-9]{64}$')


class EnrollmentInput(Input):
    upload_id: str=Field(pattern='^[a-f0-9]{32}$')
    name: str=Field(min_length=1,max_length=80)


class ApprovalInput(Input):
    revision: str=Field(pattern='^[a-f0-9]{64}$')
    accept_static_reference_limitations: Literal[True]


def asset_view(row,rev=None):
    return {'id':row.id,'name':row.name,'status':row.status,'revision':row.revision,
        'consent_id':row.consent_id,'visual_mode':'static_mesh_reference','voice_kind':'existing_synthetic_fixture',
        'evaluation':rev.evaluation if rev else None,'client_distribution':False}


def job_view(row):
    return {'id':row.id,'asset_id':row.asset_id,'status':row.status,'stage':row.stage,
            'attempt':row.attempt,'error_code':row.error_code}


def routes(service,principal,actors,mutation_lock):
    router=APIRouter(prefix='/v1')
    async def call(function,*args,**kwargs):
        try: return await asyncio.to_thread(function,*args,**kwargs)
        except LookupError: raise HTTPException(404,'not_found')
        except PermissionError: raise HTTPException(403,'authorization_unavailable')
        except ValueError as exc: raise HTTPException(409,str(exc))
        except RuntimeError: raise HTTPException(429,'resource_limit')
    def key(request): return request.headers.get('idempotency-key','')

    @router.post('/consents',status_code=201)
    async def consent(body: ConsentInput,request: Request):
        p=await principal(request,mutation=True)
        row=await call(service.create_consent,p.id,key(request),body.model_dump(exclude={'attested'}))
        return {'id':row.id,'kind':row.kind,'status':row.status,'allowed_uses':row.allowed_uses}

    @router.post('/uploads',status_code=201)
    async def upload(body: UploadInput,request: Request):
        p=await principal(request,mutation=True)
        row,token=await call(service.create_upload,p.id,key(request),body.model_dump(exclude={'purpose','content_type'}))
        return {'id':row.id,'status':row.status,'write_token':token,'expires_at':row.expires.isoformat(),
                'content_url':f'/v1/uploads/{row.id}/content'}

    @router.put('/uploads/{rid}/content')
    async def content(rid: str,request: Request):
        p=await principal(request,mutation=True)
        if request.headers.get('content-type','').split(';')[0]!='application/json': raise HTTPException(415,'json_bundle_required')
        data=await request.body()
        row=await call(service.put_upload,p.id,rid,request.headers.get('x-upload-token',''),data)
        return {'id':row.id,'status':row.status}

    @router.post('/uploads/{rid}/complete')
    async def complete(rid: str,request: Request):
        p=await principal(request,mutation=True)
        row=await call(service.get,Upload,p.id,rid)
        if row.status!='sealed': raise HTTPException(409,'upload_not_sealed')
        return {'id':row.id,'status':row.status}

    @router.post('/enrollments',status_code=202)
    async def enroll(body: EnrollmentInput,request: Request):
        p=await principal(request,mutation=True)
        return job_view(await call(service.enroll,p.id,key(request),body.model_dump()))

    @router.get('/jobs/{rid}')
    async def job(rid: str,request: Request):
        p=await principal(request)
        result=job_view(await call(service.get,Job,p.id,rid))
        result['attempts']=await call(service.history,p.id,rid)
        return result

    @router.get('/jobs')
    async def jobs(request: Request,after: str=Query('',pattern='^([a-f0-9]{32})?$'),limit: int=Query(50,ge=1,le=50)):
        p=await principal(request)
        rows=await call(service.rows,Job,p.id,after,limit)
        return {'items':[job_view(row) for row in rows],'next_cursor':rows[-1].id if len(rows)==limit else None}

    @router.post('/jobs/{rid}/cancel')
    async def cancel(rid: str,request: Request):
        p=await principal(request,mutation=True)
        return job_view(await call(service.cancel,p.id,rid))

    @router.get('/assets/{rid}')
    async def asset(rid: str,request: Request):
        p=await principal(request)
        row=await call(service.get,ManagedAsset,p.id,rid)
        rev=await call(service.get,Revision,p.id,row.revision) if row.revision else None
        return asset_view(row,rev)

    @router.get('/assets/{rid}/preview')
    async def preview(rid: str,request: Request):
        p=await principal(request)
        row=await call(service.get,ManagedAsset,p.id,rid)
        await call(service.get,Consent,p.id,row.consent_id)
        if row.status not in {'review','approved'} or not row.revision: raise HTTPException(403,'asset_unavailable')
        rev=await call(service.get,Revision,p.id,row.revision)
        path=service.root/rev.object_key/'preview.png'
        if path.is_symlink() or not path.is_file(): raise HTTPException(404,'preview_unavailable')
        return Response(await asyncio.to_thread(path.read_bytes),media_type='image/png',headers={'Cache-Control':'no-store'})

    @router.post('/assets/{rid}/approve')
    async def approve(rid: str,body: ApprovalInput,request: Request):
        p=await principal(request,mutation=True)
        row=await call(service.approve,p.id,rid,key(request),body.model_dump())
        return asset_view(row)

    async def revoke_resource(rid,request,is_consent):
        p=await principal(request,mutation=True)
        async with mutation_lock:
            ids=await call(service.revoke,p.id,rid,consent=is_consent)
            # Fence before the first cleanup await, including in-flight publication.
            for actor in actors.values():
                if actor.row.tenant_id==p.id and actor.row.asset_id in ids:
                    actor.media.fence(permanent=True)
        return {'status':'revoked','affected_assets':ids,'cleanup':'pending'}

    @router.post('/assets/{rid}/revoke',status_code=202)
    async def revoke(rid: str,request: Request): return await revoke_resource(rid,request,False)

    @router.post('/consents/{rid}/revoke',status_code=202)
    async def revoke_consent(rid: str,request: Request): return await revoke_resource(rid,request,True)

    @router.get('/assets/{rid}/deletion')
    async def receipt(rid: str,request: Request):
        p=await principal(request)
        row=await call(service.get,ManagedAsset,p.id,rid)
        if row.status!='revoked': raise HTTPException(409,'asset_not_revoked')
        pending=await call(service.deletion_pending,p.id,rid)
        return {'asset_id':rid,'status':'pending' if pending else 'complete','scope':'owned_import_and_derived_objects',
                'backup_tombstone_replay_qualified':False,'client_distribution':False}

    return router


async def worker(service,actors,lk):
    while True:
        try:
            claim=await asyncio.to_thread(service.claim)
            if claim:
                try:
                    pending=asyncio.create_task(asyncio.to_thread(validate_import,service,claim))
                    try: await asyncio.shield(pending)
                    except asyncio.CancelledError:
                        await asyncio.gather(pending,return_exceptions=True);raise
                except PermissionError: pass  # Cancel/revoke/lease fencing won the commit.
                except (ValueError,TypeError,KeyError,ContractError): await asyncio.to_thread(service.fail,claim,'invalid_import')
                except OSError: await asyncio.to_thread(service.fail,claim,'storage_temporarily_unavailable',retry=True)
                except Exception:
                    logging.getLogger(__name__).warning('canonical import failed')
                    await asyncio.to_thread(service.fail,claim,'import_failed')
            for event in await asyncio.to_thread(service.outbox):
                try:
                    if event.kind=='revoke_consent':
                        await asyncio.to_thread(service.purge_consent,event);continue
                    for row in await asyncio.to_thread(service.affected_sessions,event):
                        actor=actors.get(row.id)
                        if actor:
                            actor.media.fence(permanent=True)
                            if await actor.close(error='asset_revoked') is False: raise RuntimeError('close_pending')
                        try: await lk.room.delete_room(api.DeleteRoomRequest(room=row.room))
                        except Exception as exc:
                            if getattr(exc,'code',None)!='not_found': raise
                        await asyncio.to_thread(service.stop_orphan,row.id)
                    await asyncio.to_thread(service.purge_and_receipt,event)
                except Exception: logging.getLogger(__name__).warning('revocation cleanup pending')
            await asyncio.to_thread(service.expire_uploads)
        except asyncio.CancelledError: raise
        except Exception: logging.getLogger(__name__).warning('lifecycle worker retrying')
        await asyncio.sleep(.25)
