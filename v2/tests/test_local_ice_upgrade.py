"""Exercise the old-to-new local profile migration without Docker or models."""
import importlib.util
from pathlib import Path


def test_ice_migration_preserves_private_config_and_is_idempotent(tmp_path, monkeypatch):
    spec=importlib.util.spec_from_file_location('p04_local',Path(__file__).parents[1]/'scripts/p04_local.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module,'PRIVATE',tmp_path)
    path=tmp_path/'livekit.yaml'
    old='port: 7880\nrtc:\n  node_ip: 127.0.0.1\n  use_external_ip: false\nkeys:\n  fixture: keep-this-secret\n'
    path.write_text(old)
    assert module.upgrade_local_ice()
    assert path.read_text()==old.replace('  node_ip: 127.0.0.1\n','')
    assert path.stat().st_mode & 0o777==0o600
    assert not module.upgrade_local_ice()
    custom=old.replace('127.0.0.1','10.1.2.3')
    path.write_text(custom)
    assert not module.upgrade_local_ice()
    assert path.read_text()==custom
