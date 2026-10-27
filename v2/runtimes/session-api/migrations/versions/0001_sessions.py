"""Initial PostgreSQL session/asset metadata. No biometric media or transcripts."""
from alembic import op
import sqlalchemy as sa

revision='p04_0001'
down_revision=None

def upgrade():
    op.create_table('principals',sa.Column('id',sa.String(32),primary_key=True),
        sa.Column('cookie_hash',sa.String(64),nullable=False,unique=True),sa.Column('csrf_hash',sa.String(64),nullable=False),
        sa.Column('expires',sa.DateTime(timezone=True),nullable=False))
    op.create_table('asset_revisions',sa.Column('tenant_id',sa.String(32),primary_key=True),sa.Column('id',sa.String(80),primary_key=True),
        sa.Column('revision',sa.String(64),nullable=False),sa.Column('status',sa.String(20),nullable=False))
    op.create_table('sessions',sa.Column('id',sa.String(32),primary_key=True),sa.Column('tenant_id',sa.String(32),nullable=False),
        sa.Column('room',sa.String(80),nullable=False,unique=True),sa.Column('asset_id',sa.String(80),nullable=False),
        sa.Column('asset_revision',sa.String(64),nullable=False),sa.Column('status',sa.String(20),nullable=False),
        sa.Column('epoch',sa.Integer,nullable=False),sa.Column('fence',sa.Integer,nullable=False),
        sa.Column('lease_expires',sa.DateTime(timezone=True),nullable=False),sa.Column('created',sa.DateTime(timezone=True),nullable=False),
        sa.Column('error_code',sa.String(60),nullable=True))
    op.create_index('ix_sessions_tenant_id','sessions',['tenant_id']);op.create_index('ix_sessions_status','sessions',['status'])
    op.create_table('idempotency_records',sa.Column('tenant_id',sa.String(32),primary_key=True),sa.Column('key',sa.String(100),primary_key=True),
        sa.Column('body_hash',sa.String(64),nullable=False),sa.Column('session_id',sa.String(32),nullable=False),
        sa.Column('expires',sa.DateTime(timezone=True),nullable=False))
    op.create_table('session_controls',sa.Column('session_id',sa.String(32),primary_key=True),sa.Column('sequence',sa.Integer,primary_key=True),
        sa.Column('body_hash',sa.String(64),nullable=False),sa.Column('result',sa.JSON,nullable=False))

def downgrade():
    for name in ['session_controls','idempotency_records','sessions','asset_revisions','principals']: op.drop_table(name)
