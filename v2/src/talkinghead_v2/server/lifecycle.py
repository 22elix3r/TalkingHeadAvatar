"""Durable tenant-scoped import/review/revocation, separate from live media.

Only bounded canonical JSON meshes are supported. No training, vendor deserializer,
model download or claim of likeness/lip-sync quality is made by this importer.
"""
from contextlib import contextmanager
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil

from sqlalchemy import String,Integer,DateTime,JSON,select,text
from sqlalchemy.orm import Mapped,mapped_column

from talkinghead_v2.server.db import Base,Asset,Session,now,opaque,hashed

MAX_UPLOAD=4*1024*1024


class Consent(Base):
    __tablename__='consent_records'
    id: Mapped[str]=mapped_column(String(32),primary_key=True)
    tenant_id: Mapped[str]=mapped_column(String(32),index=True)
    kind: Mapped[str]=mapped_column(String(30))
    reference: Mapped[str]=mapped_column(String(300))
    allowed_uses: Mapped[list]=mapped_column(JSON)
    status: Mapped[str]=mapped_column(String(20),default='active')
    generation: Mapped[int]=mapped_column(Integer,default=1)
    created: Mapped[object]=mapped_column(DateTime(timezone=True),default=now)


class Upload(Base):
    __tablename__='uploads'
    id: Mapped[str]=mapped_column(String(32),primary_key=True)
    tenant_id: Mapped[str]=mapped_column(String(32),index=True)
    consent_id: Mapped[str]=mapped_column(String(32))
    size: Mapped[int]=mapped_column(Integer)
    sha256: Mapped[str]=mapped_column(String(64))
    token_hash: Mapped[str]=mapped_column(String(64))
    status: Mapped[str]=mapped_column(String(20),default='pending')
    expires: Mapped[object]=mapped_column(DateTime(timezone=True))


class ManagedAsset(Base):
    __tablename__='managed_assets'
    id: Mapped[str]=mapped_column(String(32),primary_key=True)
    tenant_id: Mapped[str]=mapped_column(String(32),index=True)
    consent_id: Mapped[str]=mapped_column(String(32))
    name: Mapped[str]=mapped_column(String(80))
    status: Mapped[str]=mapped_column(String(20),default='importing')
    generation: Mapped[int]=mapped_column(Integer,default=1)
    revision: Mapped[str|None]=mapped_column(String(64),nullable=True)
    created: Mapped[object]=mapped_column(DateTime(timezone=True),default=now)


class Revision(Base):
    __tablename__='managed_revisions'
    id: Mapped[str]=mapped_column(String(64),primary_key=True)
    tenant_id: Mapped[str]=mapped_column(String(32),index=True)
    asset_id: Mapped[str]=mapped_column(String(32),index=True)
    upload_id: Mapped[str]=mapped_column(String(32))
    object_key: Mapped[str]=mapped_column(String(140))
    evaluation: Mapped[dict]=mapped_column(JSON)
    tts_revision: Mapped[str]=mapped_column(String(64))
    created: Mapped[object]=mapped_column(DateTime(timezone=True),default=now)


class Job(Base):
    __tablename__='jobs'
    id: Mapped[str]=mapped_column(String(32),primary_key=True)
    tenant_id: Mapped[str]=mapped_column(String(32),index=True)
    asset_id: Mapped[str]=mapped_column(String(32),unique=True)
    upload_id: Mapped[str]=mapped_column(String(32))
    status: Mapped[str]=mapped_column(String(20),default='queued')
    stage: Mapped[str]=mapped_column(String(20),default='validate')
    attempt: Mapped[int]=mapped_column(Integer,default=0)
    fence: Mapped[int]=mapped_column(Integer,default=0)
    generation: Mapped[int]=mapped_column(Integer)
    consent_generation: Mapped[int]=mapped_column(Integer)
    lease_expires: Mapped[object]=mapped_column(DateTime(timezone=True),default=now)
    available_after: Mapped[object]=mapped_column(DateTime(timezone=True),default=now)
    error_code: Mapped[str|None]=mapped_column(String(60),nullable=True)
    created: Mapped[object]=mapped_column(DateTime(timezone=True),default=now)


class Operation(Base):
    __tablename__='lifecycle_operations'
    tenant_id: Mapped[str]=mapped_column(String(32),primary_key=True)
    scope: Mapped[str]=mapped_column(String(70),primary_key=True)
    key: Mapped[str]=mapped_column(String(100),primary_key=True)
    body_hash: Mapped[str]=mapped_column(String(64))
    resource_id: Mapped[str]=mapped_column(String(64))
    expires: Mapped[object]=mapped_column(DateTime(timezone=True))


class JobAttempt(Base):
    __tablename__='job_attempts'
    job_id: Mapped[str]=mapped_column(String(32),primary_key=True)
    fence: Mapped[int]=mapped_column(Integer,primary_key=True)
    tenant_id: Mapped[str]=mapped_column(String(32),index=True)
    attempt: Mapped[int]=mapped_column(Integer)
    stage: Mapped[str]=mapped_column(String(20))
    status: Mapped[str]=mapped_column(String(20))
    started: Mapped[object]=mapped_column(DateTime(timezone=True),default=now)
    finished: Mapped[object|None]=mapped_column(DateTime(timezone=True),nullable=True)
    error_code: Mapped[str|None]=mapped_column(String(60),nullable=True)


class Outbox(Base):
    __tablename__='outbox_events'
    id: Mapped[str]=mapped_column(String(32),primary_key=True)
    tenant_id: Mapped[str]=mapped_column(String(32),index=True)
    kind: Mapped[str]=mapped_column(String(30))
    resource_id: Mapped[str]=mapped_column(String(32))
    status: Mapped[str]=mapped_column(String(20),default='pending')
    created: Mapped[object]=mapped_column(DateTime(timezone=True),default=now)


class Audit(Base):
    __tablename__='audit_events'
    id: Mapped[str]=mapped_column(String(32),primary_key=True)
    tenant_id: Mapped[str]=mapped_column(String(32),index=True)
    action: Mapped[str]=mapped_column(String(40))
    resource_id: Mapped[str]=mapped_column(String(64))
    created: Mapped[object]=mapped_column(DateTime(timezone=True),default=now)


def strict_json(data):
    def pairs(rows):
        result={}
        for key,value in rows:
            if key in result: raise ValueError('duplicate_json_key')
            result[key]=value
        return result
    def invalid(value): raise ValueError('nonfinite_json')
    return json.loads(data,object_pairs_hook=pairs,parse_constant=invalid)


class Lifecycle:
    def __init__(self,store,root: Path,tts_revision):
        self.store,self.root,self.tts_revision=store,root,tts_revision
        root.mkdir(parents=True,exist_ok=True,mode=0o700)
        os.chmod(root,0o700)

    @contextmanager
    def transaction(self):
        with self.store.sessions.begin() as db:
            # Serializes short lifecycle commits, never model/render work.
            db.execute(text('SELECT pg_advisory_xact_lock(1740290405)'))
            yield db

    def owned(self,db,model,tenant,rid):
        row=db.get(model,rid)
        if row is None or row.tenant_id!=tenant: raise LookupError('not_found')
        return row

    def consent(self,db,tenant,rid):
        row=self.owned(db,Consent,tenant,rid)
        if row.status!='active': raise PermissionError('consent_revoked')
        return row

    def audit(self,db,tenant,action,rid): db.add(Audit(id=opaque(),tenant_id=tenant,action=action,resource_id=rid))

    def operation(self,db,tenant,scope,key,body):
        if not 1<=len(key)<=100: raise ValueError('idempotency_key_required')
        digest=hashed(json.dumps(body,sort_keys=True,separators=(',',':')))
        old=db.get(Operation,(tenant,scope,key))
        if old and old.expires>now():
            if old.body_hash!=digest: raise ValueError('idempotency_conflict')
            return old.resource_id,None
        if old: db.delete(old);db.flush()
        record=Operation(tenant_id=tenant,scope=scope,key=key,body_hash=digest,resource_id='',expires=now()+timedelta(hours=24))
        db.add(record)
        return None,record

    def create_consent(self,tenant,key,body):
        with self.transaction() as db:
            old,op=self.operation(db,tenant,'consent',key,body)
            if old: return self.owned(db,Consent,tenant,old)
            row=Consent(id=opaque(),tenant_id=tenant,**body)
            db.add(row);op.resource_id=row.id;self.audit(db,tenant,'consent_recorded',row.id)
            db.flush();return row

    def create_upload(self,tenant,key,body):
        with self.transaction() as db:
            self.consent(db,tenant,body['consent_id'])
            old,op=self.operation(db,tenant,'upload',key,body)
            if old:
                row=self.owned(db,Upload,tenant,old)
                # Rotate the write grant on a retry; grants are never stored raw.
                token=secrets.token_urlsafe(32);row.token_hash=hashed(token)
                return row,token
            if db.scalar(select(Upload.id).where(Upload.tenant_id==tenant,Upload.status.in_(['pending','sealed']),Upload.expires>now()).offset(7).limit(1)):
                raise RuntimeError('upload_limit')
            token=secrets.token_urlsafe(32)
            row=Upload(id=opaque(),tenant_id=tenant,**body,token_hash=hashed(token),expires=now()+timedelta(minutes=15))
            db.add(row);op.resource_id=row.id;db.flush();return row,token

    def upload_path(self,tenant,rid): return self.root/tenant/'uploads'/rid

    def write_object(self,path,data):
        path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        if path.exists() or path.is_symlink():
            if path.is_symlink() or path.read_bytes()!=data: raise ValueError('immutable_object_conflict')
            return
        temporary=path.parent/('.pending-'+opaque())
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        try:
            with os.fdopen(fd,'wb') as stream: stream.write(data);stream.flush();os.fsync(stream.fileno())
            os.chmod(temporary,0o400)
            try: os.link(temporary,path,follow_symlinks=False)
            except FileExistsError:
                if path.is_symlink() or path.read_bytes()!=data: raise ValueError('immutable_object_conflict')
            fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
        finally: temporary.unlink(missing_ok=True)

    def put_upload(self,tenant,rid,token,data):
        with self.transaction() as db:
            row=self.owned(db,Upload,tenant,rid);self.consent(db,tenant,row.consent_id)
            if row.expires<=now() or not secrets.compare_digest(row.token_hash,hashed(token)): raise PermissionError('upload_grant_invalid')
            if row.status not in {'pending','sealed'}: raise ValueError('upload_closed')
            if len(data)!=row.size or hashlib.sha256(data).hexdigest()!=row.sha256: raise ValueError('upload_size_or_hash_mismatch')
            self.write_object(self.upload_path(tenant,rid),data)
            row.status='sealed';return row

    def enroll(self,tenant,key,body):
        with self.transaction() as db:
            old,op=self.operation(db,tenant,'enrollment',key,body)
            if old: return self.owned(db,Job,tenant,old)
            upload=self.owned(db,Upload,tenant,body['upload_id']);consent=self.consent(db,tenant,upload.consent_id)
            if upload.status!='sealed' or upload.expires<=now(): raise ValueError('upload_not_sealed')
            if db.scalar(select(ManagedAsset.id).where(ManagedAsset.tenant_id==tenant,ManagedAsset.status!='revoked').offset(31).limit(1)):
                raise RuntimeError('asset_limit')
            asset=ManagedAsset(id=opaque(),tenant_id=tenant,consent_id=consent.id,name=body['name'])
            db.add(asset);db.flush()
            job=Job(id=opaque(),tenant_id=tenant,asset_id=asset.id,upload_id=upload.id,generation=asset.generation,consent_generation=consent.generation)
            db.add(job);upload.status='consumed';op.resource_id=job.id
            self.audit(db,tenant,'import_queued',asset.id);db.flush();return job

    def claim(self):
        with self.transaction() as db:
            row=db.scalar(select(Job).where(((Job.status=='queued')&(Job.available_after<=now()))|((Job.status=='running')&(Job.lease_expires<now())))
                          .order_by(Job.created).with_for_update(skip_locked=True).limit(1))
            if row is None: return None
            asset=self.owned(db,ManagedAsset,row.tenant_id,row.asset_id)
            consent=self.owned(db,Consent,row.tenant_id,asset.consent_id)
            previous=db.get(JobAttempt,(row.id,row.fence))
            if previous and previous.status=='running': previous.status='expired';previous.finished=now()
            if asset.status=='revoked' or consent.status!='active' or row.attempt>=3:
                row.status='cancelled' if asset.status=='revoked' or consent.status!='active' else 'failed'
                row.error_code='authorization_lost' if row.status=='cancelled' else 'attempts_exhausted'
                if row.status=='failed': asset.status='failed'
                return None
            row.status='running';row.attempt+=1;row.fence+=1;row.lease_expires=now()+timedelta(seconds=30)
            db.add(JobAttempt(job_id=row.id,fence=row.fence,tenant_id=row.tenant_id,attempt=row.attempt,stage='validate',status='running'))
            db.flush();return row,asset,consent

    def stage(self,claim,stage):
        job,*_=claim
        with self.transaction() as db:
            row=self.owned(db,Job,job.tenant_id,job.id)
            if row.status!='running' or row.fence!=job.fence or row.lease_expires<=now(): raise PermissionError('stale_import_attempt')
            row.stage=stage;row.lease_expires=now()+timedelta(seconds=30)
            db.get(JobAttempt,(row.id,row.fence)).stage=stage

    def finish(self,claim,revision,evaluation):
        job,asset,consent=claim
        with self.transaction() as db:
            current=self.owned(db,Job,job.tenant_id,job.id)
            a=self.owned(db,ManagedAsset,asset.tenant_id,asset.id);c=self.owned(db,Consent,consent.tenant_id,consent.id)
            if (current.status!='running' or current.fence!=job.fence or current.lease_expires<=now() or
                a.generation!=job.generation or c.generation!=job.consent_generation or c.status!='active' or a.status=='revoked'):
                raise PermissionError('stale_import_attempt')
            existing=db.get(Revision,revision)
            if existing is None:
                db.add(Revision(id=revision,tenant_id=a.tenant_id,asset_id=a.id,upload_id=job.upload_id,
                    object_key=f'{a.tenant_id}/assets/{a.id}/{revision}',evaluation=evaluation,tts_revision=self.tts_revision))
            a.status='review';a.revision=revision;current.status='review';current.stage='review'
            attempt=db.get(JobAttempt,(job.id,job.fence));attempt.status='complete';attempt.stage='review';attempt.finished=now()
            self.audit(db,a.tenant_id,'import_ready_for_review',a.id)

    def write_import_object(self,claim,path,data):
        job,asset,consent=claim
        with self.transaction() as db:
            current=self.owned(db,Job,job.tenant_id,job.id)
            a=self.owned(db,ManagedAsset,asset.tenant_id,asset.id);c=self.owned(db,Consent,consent.tenant_id,consent.id)
            if (current.status!='running' or current.fence!=job.fence or current.lease_expires<=now() or
                a.generation!=job.generation or c.generation!=job.consent_generation or c.status!='active' or a.status=='revoked'):
                raise PermissionError('stale_import_attempt')
            self.write_object(path,data)

    def fail(self,claim,code,*,retry=False):
        job,*_=claim
        with self.transaction() as db:
            row=self.owned(db,Job,job.tenant_id,job.id)
            if row.status=='running' and row.fence==job.fence:
                row.status='queued' if retry and row.attempt<3 else 'failed';row.error_code=code
                row.available_after=now()+timedelta(seconds=2**row.attempt)
                attempt=db.get(JobAttempt,(job.id,job.fence));attempt.status='failed';attempt.error_code=code;attempt.finished=now()
                if row.status=='failed':
                    asset=self.owned(db,ManagedAsset,job.tenant_id,job.asset_id)
                    if asset.generation==job.generation: asset.status='failed'

    def approve(self,tenant,rid,key,body):
        with self.transaction() as db:
            asset=self.owned(db,ManagedAsset,tenant,rid);self.consent(db,tenant,asset.consent_id)
            if asset.status=='revoked': raise PermissionError('asset_revoked')
            old,op=self.operation(db,tenant,'approve:'+rid,key,body)
            if old: return asset
            if asset.status!='review' or asset.revision!=body['revision']: raise ValueError('review_revision_changed')
            rev=self.owned(db,Revision,tenant,asset.revision)
            if rev.tts_revision!=self.tts_revision: raise ValueError('tts_profile_changed')
            asset.status='approved'
            db.add(Asset(tenant_id=tenant,id=asset.id,revision=asset.revision,status='approved'))
            job=db.scalar(select(Job).where(Job.asset_id==asset.id));job.status='complete';job.stage='published'
            op.resource_id=asset.id;self.audit(db,tenant,'asset_approved',asset.id)
            return asset

    def cancel(self,tenant,rid):
        with self.transaction() as db:
            job=self.owned(db,Job,tenant,rid)
            if job.status=='complete': raise ValueError('already_published_revoke_asset_instead')
            if job.status!='cancelled':
                attempt=db.get(JobAttempt,(job.id,job.fence))
                if attempt and attempt.status=='running': attempt.status='cancelled';attempt.finished=now()
                job.status='cancelled';job.fence+=1
                asset=self.owned(db,ManagedAsset,tenant,job.asset_id)
                asset.generation+=1;asset.status='cancelled'
            return job

    def revoke(self,tenant,rid,*,consent=False):
        with self.transaction() as db:
            if consent:
                record=self.owned(db,Consent,tenant,rid)
                if record.status=='active':
                    record.status='revoked';record.generation+=1;self.audit(db,tenant,'consent_revoked',rid)
                    db.add(Outbox(id=opaque(),tenant_id=tenant,kind='revoke_consent',resource_id=rid))
                assets=list(db.scalars(select(ManagedAsset).where(ManagedAsset.tenant_id==tenant,ManagedAsset.consent_id==rid)))
            else: assets=[self.owned(db,ManagedAsset,tenant,rid)]
            ids=[]
            for asset in assets:
                ids.append(asset.id)
                if asset.status=='revoked': continue
                asset.status='revoked';asset.generation+=1
                serving=db.get(Asset,(tenant,asset.id),with_for_update=True)
                if serving: serving.status='revoked'
                for job in db.scalars(select(Job).where(Job.asset_id==asset.id,Job.status.in_(['queued','running','review']))):
                    attempt=db.get(JobAttempt,(job.id,job.fence))
                    if attempt and attempt.status=='running': attempt.status='cancelled';attempt.finished=now()
                    job.status='cancelled';job.fence+=1;job.error_code='authorization_lost'
                for session in db.scalars(select(Session).where(Session.tenant_id==tenant,Session.asset_id==asset.id,
                    Session.status.in_(['PREPARING','READY','ACTIVE']))):
                    session.status='DRAINING';session.error_code='asset_revoked'
                db.add(Outbox(id=opaque(),tenant_id=tenant,kind='revoke_asset',resource_id=asset.id))
                self.audit(db,tenant,'asset_revoked',asset.id)
            return ids

    def rows(self,model,tenant,after='',limit=100):
        with self.store.sessions() as db:
            query=select(model).where(model.tenant_id==tenant)
            if after: query=query.where(model.id>after)
            return list(db.scalars(query.order_by(model.id).limit(limit)))

    def get(self,model,tenant,rid):
        with self.store.sessions() as db: return self.owned(db,model,tenant,rid)

    def serving_manifest(self,tenant,rid,revision):
        with self.store.sessions() as db:
            asset=self.owned(db,ManagedAsset,tenant,rid);self.consent(db,tenant,asset.consent_id)
            if asset.status!='approved' or asset.revision!=revision: raise PermissionError('asset_unavailable')
            rev=self.owned(db,Revision,tenant,revision)
            if rev.tts_revision!=self.tts_revision: raise PermissionError('tts_profile_changed')
            path=self.root/rev.object_key/'avatar.json'
            if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_UPLOAD or hashlib.sha256(path.read_bytes()).hexdigest()!=rev.evaluation['manifest_sha256']:
                raise PermissionError('immutable_manifest_changed')
            return path

    def history(self,tenant,rid):
        with self.store.sessions() as db:
            self.owned(db,Job,tenant,rid)
            rows=db.scalars(select(JobAttempt).where(JobAttempt.tenant_id==tenant,JobAttempt.job_id==rid).order_by(JobAttempt.attempt))
            return [{'attempt':r.attempt,'fence':r.fence,'stage':r.stage,'status':r.status,'started_at':r.started.isoformat(),
                     'finished_at':r.finished.isoformat() if r.finished else None,'error_code':r.error_code} for r in rows]

    def outbox(self):
        with self.store.sessions() as db:
            return list(db.scalars(select(Outbox).where(Outbox.status=='pending').order_by(Outbox.created).limit(20)))

    def affected_sessions(self,event):
        with self.store.sessions() as db:
            return list(db.scalars(select(Session).where(Session.tenant_id==event.tenant_id,Session.asset_id==event.resource_id)))

    def purge_and_receipt(self,event):
        with self.transaction() as db:
            current=db.get(Outbox,event.id)
            if current.status=='complete': return
            asset=self.owned(db,ManagedAsset,event.tenant_id,event.resource_id)
            if asset.status!='revoked': raise PermissionError('revocation_tombstone_required')
            active=db.scalar(select(Session.id).where(Session.tenant_id==event.tenant_id,Session.asset_id==asset.id,
                Session.status.in_(['PREPARING','READY','ACTIVE','DRAINING'])).limit(1))
            if active: raise RuntimeError('session_cleanup_pending')
            directory=self.root/event.tenant_id/'assets'/asset.id
            if directory.exists(): shutil.rmtree(directory)
            for job in db.scalars(select(Job).where(Job.asset_id==asset.id)):
                self.upload_path(event.tenant_id,job.upload_id).unlink(missing_ok=True)
            current.status='complete';self.audit(db,event.tenant_id,'derived_objects_purged',asset.id)

    def deletion_pending(self,tenant,rid):
        with self.store.sessions() as db:
            return db.scalar(select(Outbox.id).where(Outbox.tenant_id==tenant,Outbox.resource_id==rid,Outbox.status!='complete').limit(1)) is not None

    def stop_orphan(self,sid):
        with self.transaction() as db:
            row=db.get(Session,sid,with_for_update=True)
            if row.status=='DRAINING': row.status='CLOSED';row.fence+=1;row.error_code='asset_revoked'

    def purge_consent(self,event):
        with self.transaction() as db:
            consent=self.owned(db,Consent,event.tenant_id,event.resource_id)
            if consent.status!='revoked': raise PermissionError('consent_tombstone_required')
            for asset in db.scalars(select(ManagedAsset).where(ManagedAsset.tenant_id==event.tenant_id,ManagedAsset.consent_id==consent.id)):
                pending=db.scalar(select(Outbox.id).where(Outbox.tenant_id==event.tenant_id,Outbox.resource_id==asset.id,Outbox.status!='complete').limit(1))
                if pending: return False
            for upload in db.scalars(select(Upload).where(Upload.tenant_id==event.tenant_id,Upload.consent_id==consent.id)):
                self.upload_path(event.tenant_id,upload.id).unlink(missing_ok=True);upload.status='revoked'
            db.get(Outbox,event.id).status='complete'
            self.audit(db,event.tenant_id,'consent_objects_purged',consent.id)

    def expire_uploads(self):
        with self.transaction() as db:
            for upload in db.scalars(select(Upload).where(Upload.status.in_(['pending','sealed']),Upload.expires<now()).limit(20)):
                self.upload_path(upload.tenant_id,upload.id).unlink(missing_ok=True);upload.status='expired'
