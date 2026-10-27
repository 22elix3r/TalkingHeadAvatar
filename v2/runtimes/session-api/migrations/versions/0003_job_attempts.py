"""Persist import attempt/fence history without fabricating historical attempts."""
from alembic import op
import sqlalchemy as sa

revision='p05_0003'
down_revision='p05_0002'


def upgrade():
    op.create_table('job_attempts',
        sa.Column('job_id',sa.String(32),primary_key=True),sa.Column('fence',sa.Integer,primary_key=True),
        sa.Column('tenant_id',sa.String(32),nullable=False),sa.Column('attempt',sa.Integer,nullable=False),
        sa.Column('stage',sa.String(20),nullable=False),sa.Column('status',sa.String(20),nullable=False),
        sa.Column('started',sa.DateTime(timezone=True),nullable=False),
        sa.Column('finished',sa.DateTime(timezone=True),nullable=True),sa.Column('error_code',sa.String(60),nullable=True))
    op.create_index('ix_job_attempts_tenant_id','job_attempts',['tenant_id'])


def downgrade():
    op.drop_index('ix_job_attempts_tenant_id',table_name='job_attempts')
    op.drop_table('job_attempts')
