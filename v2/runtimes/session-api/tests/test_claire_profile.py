"""Real PostgreSQL/auth/profile boundaries; inference is not run by these tests."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from talkinghead_v2.server import app as module
from talkinghead_v2.server.db import Asset, hashed

from test_control_api import settings  # Dedicated tha_p04_test reset/schema fixture.

ASSET = 'claire-vendor-research'
PROFILE = 'claire_vendor_research'
REVISION = hashed('f' * 64 + ':' + 'd' * 64)


@pytest.fixture
async def client(settings, monkeypatch):
    class FakeActor:
        def __init__(self, config, store, row, *providers):
            self.store, self.row, self.epoch, self.closed = store, row, 0, False
            self.control_lock = asyncio.Lock()
            self.media = SimpleNamespace(fence=Mock())
        async def prepare(self): self.store.update(self.row.id, status='READY')
        async def close(self, **kwargs):
            self.closed = True
            self.store.update(self.row.id, status='CLOSED')
            return True
    monkeypatch.setattr(module, 'Actor', FakeActor)
    config = replace(settings, media_backend='native_claire_research', native_media_runtime=settings.private_dir/'operator.json',
                     origin='http://127.0.0.1:18730')
    app = module.create_app(config, load_models=False, run_jobs=False)
    async with app.router.lifespan_context(app):
        app.state.resources.update(ready=True, config=None, asr=None, llm=None, tts_revision='f' * 64, asset_revision=REVISION,
            avatar_config=SimpleNamespace(revision='d' * 64, basis_hash='a' * 64, topology_hash='b' * 64, model_hash='c' * 64))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=('127.0.0.1', 123)), base_url=config.origin) as c:
            c.headers['Origin'] = config.origin
            c.cookies.set('tha_dev', 'normal-preview-cookie')
            c.cookies.set('tha_csrf', 'normal-preview-csrf')
            login = await c.post('/v1/dev-login', json={})
            assert login.status_code == 200
            c.headers['X-CSRF-Token'] = login.json()['csrf_token']
            yield c, app, login.json()


async def test_research_cookie_scope_and_hash_bound_builtin_are_distinct(client):
    c, app, login = client
    assert login['profile'] == PROFILE and login['default_asset_id'] == ASSET
    assert c.cookies.get('tha_dev') == 'normal-preview-cookie'
    assert c.cookies.get('tha_csrf') == 'normal-preview-csrf'
    assert c.cookies.get('tha_dev_claire') and c.cookies.get('tha_csrf_claire')
    asset = (await c.get('/v1/assets')).json()['items'][0]
    assert asset['id'] == ASSET and asset['revision'] == REVISION and asset['status'] == 'approved'
    assert asset['basis_hash'] == 'a' * 64 and asset['topology_hash'] == 'b' * 64 and asset['model_hash'] == 'c' * 64
    assert asset['consent_evidence'] == 'vendor_assertion' and not asset['human_enrollment']
    caps = (await c.get('/v1/capabilities')).json()
    assert caps['profiles'] == [PROFILE] and not caps['asset_imports'] and not caps['visual_quality_qualified']


@pytest.mark.parametrize('profile,asset', [('audio_reference', ASSET), (PROFILE, 'synthetic-audio-reference'), (PROFILE, 'imported-other-identity')])
async def test_incompatible_profile_or_identity_is_never_substituted(client, profile, asset):
    c, _, _ = client
    result = await c.post('/v1/sessions', json={'profile': profile, 'asset_id': asset}, headers={'Idempotency-Key': 'wrong-identity'})
    assert result.status_code == 422
    assert (await c.post('/v1/consents', json={})).json()['detail'] == 'research_reference_only'


async def test_serving_revision_cannot_change_under_active_session_and_revocation_is_not_revived(client):
    c, app, login = client
    created = await c.post('/v1/sessions', json={'profile': PROFILE, 'asset_id': ASSET}, headers={'Idempotency-Key': 'vendor-session'})
    assert created.status_code == 202
    sid = created.json()['id']
    await asyncio.sleep(0)
    assert created.json()['asset_revision'] == REVISION
    assert (await c.post(f'/v1/sessions/{sid}/join-grant', json={})).status_code == 200
    with pytest.raises(RuntimeError, match='active_revision'):
        await asyncio.to_thread(app.state.store.reference_asset, login['tenant_id'], ASSET, 'e' * 64)
    assert (await c.post(f'/v1/sessions/{sid}/close', json={})).status_code == 200
    await asyncio.to_thread(app.state.store.reference_asset, login['tenant_id'], ASSET, 'e' * 64)
    with app.state.store.sessions.begin() as db:
        db.get(Asset, (login['tenant_id'], ASSET)).status = 'revoked'
    assert (await c.post('/v1/dev-login', json={})).status_code == 200
    assert (await c.get('/v1/assets')).json()['items'][0]['status'] == 'revoked'
    denied = await c.post('/v1/sessions', json={'profile': PROFILE, 'asset_id': ASSET}, headers={'Idempotency-Key': 'after-revoke'})
    assert denied.status_code == 403 and denied.json()['detail'] == 'asset_unavailable'
