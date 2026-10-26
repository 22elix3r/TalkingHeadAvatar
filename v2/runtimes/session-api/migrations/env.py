from pathlib import Path
import os
from alembic import context
from sqlalchemy import create_engine
from talkinghead_v2.server.config import Settings
from talkinghead_v2.server.db import Base
from talkinghead_v2.server import lifecycle  # Register the separately frozen P05 tables.

settings=Settings.load(Path(os.environ['THA_P04_CONFIG']))
with create_engine(settings.database_url).connect() as connection:
    context.configure(connection=connection,target_metadata=Base.metadata)
    with context.begin_transaction(): context.run_migrations()
