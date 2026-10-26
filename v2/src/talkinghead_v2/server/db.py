"""Durable local session metadata. No PCM, prompts or transcripts are stored."""

from datetime import datetime,timezone,timedelta
import hashlib
import secrets
import uuid

from sqlalchemy import String, Integer, DateTime, JSON, UniqueConstraint, create_engine, select, text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def now(): return datetime.now(timezone.utc)
def opaque(): return uuid.uuid4().hex
def hashed(value): return hashlib.sha256(value.encode()).hexdigest()


class Base(DeclarativeBase): pass


class Principal(Base):
    __tablename__='principals'
    id: Mapped[str]=mapped_column(String(32),primary_key=True)
    cookie_hash: Mapped[str]=mapped_column(String(64),unique=True)
    csrf_hash: Mapped[str]=mapped_column(String(64))
    expires: Mapped[datetime]=mapped_column(DateTime(timezone=True))


class Asset(Base):
    __tablename__='asset_revisions'
    tenant_id: Mapped[str]=mapped_column(String(32),primary_key=True)
    id: Mapped[str]=mapped_column(String(80),primary_key=True)
    revision: Mapped[str]=mapped_column(String(64))
    status: Mapped[str]=mapped_column(String(20))


class Session(Base):
    __tablename__='sessions'
    id: Mapped[str]=mapped_column(String(32),primary_key=True)
    tenant_id: Mapped[str]=mapped_column(String(32),index=True)
    room: Mapped[str]=mapped_column(String(80),unique=True)
    asset_id: Mapped[str]=mapped_column(String(80))
    asset_revision: Mapped[str]=mapped_column(String(64))
    status: Mapped[str]=mapped_column(String(20),index=True)
    epoch: Mapped[int]=mapped_column(Integer,default=0)
    fence: Mapped[int]=mapped_column(Integer,default=1)
    lease_expires: Mapped[datetime]=mapped_column(DateTime(timezone=True))
    created: Mapped[datetime]=mapped_column(DateTime(timezone=True),default=now)
    error_code: Mapped[str|None]=mapped_column(String(60),nullable=True)


class Idempotency(Base):
    __tablename__='idempotency_records'
    tenant_id: Mapped[str]=mapped_column(String(32),primary_key=True)
    key: Mapped[str]=mapped_column(String(100),primary_key=True)
    body_hash: Mapped[str]=mapped_column(String(64))
    session_id: Mapped[str]=mapped_column(String(32))
    expires: Mapped[datetime]=mapped_column(DateTime(timezone=True))


class Control(Base):
    __tablename__='session_controls'
    session_id: Mapped[str]=mapped_column(String(32),primary_key=True)
    sequence: Mapped[int]=mapped_column(Integer,primary_key=True)
    body_hash: Mapped[str]=mapped_column(String(64))
    result: Mapped[dict]=mapped_column(JSON)


class Store:
    def __init__(self,url):
        if not url.startswith('postgresql+psycopg://'): raise ValueError('PostgreSQL is required; no silent SQLite substitution')
        self.engine=create_engine(url,pool_pre_ping=True,pool_size=3,max_overflow=2)
        self.sessions=sessionmaker(self.engine,expire_on_commit=False)

    def login(self,revision):
        cookie,csrf=secrets.token_urlsafe(32),secrets.token_urlsafe(32)
        with self.sessions.begin() as db:
            principal=Principal(id=opaque(),cookie_hash=hashed(cookie),csrf_hash=hashed(csrf),expires=now()+timedelta(hours=8))
            db.add(principal)
            db.add(Asset(tenant_id=principal.id,id='synthetic-audio-reference',revision=revision,status='approved'))
        return principal.id,cookie,csrf

    def authenticate(self,cookie):
        with self.sessions() as db:
            return db.scalar(select(Principal).where(Principal.cookie_hash==hashed(cookie),Principal.expires>now()))

    def get(self,sid,tenant):
        with self.sessions() as db: return db.scalar(select(Session).where(Session.id==sid,Session.tenant_id==tenant))

    def reserve(self,tenant,key,body_hash,asset_id):
        with self.sessions.begin() as db:
            db.execute(text('SELECT pg_advisory_xact_lock(1740290404)'))
            previous=db.get(Idempotency,(tenant,key))
            if previous and previous.expires>now():
                if previous.body_hash!=body_hash: raise ValueError('idempotency_conflict')
                return db.get(Session,previous.session_id),False
            if previous: db.delete(previous);db.flush()
            asset=db.get(Asset,(tenant,asset_id))
            if not asset or asset.status!='approved': raise PermissionError('asset_unavailable')
            active=db.scalar(select(Session.id).where(Session.status.in_(['PREPARING','READY','ACTIVE'])).limit(1))
            if active: raise RuntimeError('capacity_exhausted')
            row=Session(id=opaque(),tenant_id=tenant,room='tha-p04-'+opaque(),asset_id=asset.id,
                asset_revision=asset.revision,status='PREPARING',epoch=0,fence=1,lease_expires=now()+timedelta(seconds=30))
            db.add(row);db.flush()
            db.add(Idempotency(tenant_id=tenant,key=key,body_hash=body_hash,session_id=row.id,expires=now()+timedelta(hours=24)))
            return row,True

    def update(self,sid,*,epoch=None,status=None,error=None,fence=1):
        with self.sessions.begin() as db:
            row=db.get(Session,sid,with_for_update=True)
            if row is None or row.fence!=fence: raise RuntimeError('stale_session_fence')
            if epoch is not None:
                if epoch<row.epoch: raise RuntimeError('stale_epoch')
                row.epoch=epoch
            if status: row.status=status
            row.error_code=error
            row.lease_expires=now()+timedelta(seconds=10)

    def recover(self):
        with self.sessions.begin() as db:
            rows=list(db.scalars(select(Session).where(Session.status.in_(['PREPARING','READY','ACTIVE']))))
            for row in rows:
                row.status='FAILED';row.error_code='controller_restarted';row.fence+=1
            return [(row.room,row.id) for row in rows]

    def control(self,sid,sequence):
        with self.sessions() as db: return db.get(Control,(sid,sequence))

    def last_control_sequence(self,sid):
        with self.sessions() as db:
            return db.scalar(select(func.max(Control.sequence)).where(Control.session_id==sid)) or 0

    def record_control(self,sid,sequence,body_hash,result):
        with self.sessions.begin() as db:
            db.add(Control(session_id=sid,sequence=sequence,body_hash=body_hash,result=result))
