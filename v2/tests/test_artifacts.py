import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path

import pytest

from talkinghead_v2.assets import artifacts


@pytest.fixture
def bundle(tmp_path):
    revision = "a"*40
    names = ("config.json", "model.safetensors", "speech_tokenizer/config.json",
             "speech_tokenizer/model.safetensors", "vocab.json", "merges.txt")
    bodies = {name: ("synthetic fixture for "+name).encode() for name in names}
    data = {"schema_version": 1, "model_id": artifacts.MODEL_ID, "revision": revision,
            "files": [{"path": name, "size": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                       "url": f"https://huggingface.co/{artifacts.MODEL_ID}/resolve/{revision}/{name}"}
                      for name, body in bodies.items()]}
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(data))
    return source, data, bodies


def fetcher(bundle):
    _, data, bodies = bundle
    responses = {entry["url"]: bodies[entry["path"]] for entry in data["files"]}
    return lambda url: BytesIO(responses[url])


def test_explicit_provision_checks_hashes_and_reuses_immutable_offline_snapshot(bundle, tmp_path, monkeypatch):
    source, _, _ = bundle
    manifest = artifacts.read_manifest(source)
    cache = tmp_path / "cache"
    root = artifacts.provision(cache, manifest, fetch=fetcher(bundle))
    assert artifacts.verify_snapshot(cache, manifest) == root
    assert (root / "model.safetensors").stat().st_mode & 0o222 == 0
    def forbid(*args, **kwargs):
        raise AssertionError("cache verification must not access network")
    monkeypatch.setattr(artifacts, "urlopen", forbid)
    assert artifacts.provision(cache, manifest) == root
    assert artifacts.verify_snapshot(cache, manifest) == root


@pytest.mark.parametrize("kind", ["extra", "truncated", "wrong_hash"])
def test_bad_download_never_publishes_snapshot(bundle, tmp_path, kind):
    _, manifest, _ = bundle
    normal = fetcher(bundle)
    def bad(url):
        content = normal(url).read()
        return BytesIO(content+b"extra" if kind == "extra" else content[:-1] if kind == "truncated" else b"x"*len(content))
    cache = tmp_path / "cache"
    with pytest.raises(artifacts.ArtifactError):
        artifacts.provision(cache, manifest, fetch=bad)
    assert not artifacts.snapshot_path(cache, manifest).exists()
    assert not list((cache / "blobs").glob("*.part"))


def test_corrupt_cache_is_not_silently_repaired_at_startup(bundle, tmp_path):
    _, manifest, _ = bundle
    cache = tmp_path / "cache"
    root = artifacts.provision(cache, manifest, fetch=fetcher(bundle))
    target = root / "config.json"
    target.chmod(0o644)
    target.write_bytes(b"x"*target.stat().st_size)
    with pytest.raises(artifacts.ArtifactError, match="hash mismatch"):
        artifacts.verify_snapshot(cache, manifest)
    with pytest.raises(artifacts.ArtifactError):
        artifacts.provision(cache, manifest, fetch=fetcher(bundle))


@pytest.mark.parametrize("path", ["../escape.json", "/escape.json", "a/../escape.json", "config.pkl", "config.json/", "a\\b.json"])
def test_manifest_rejects_escape_pickle_and_noncanonical_paths(bundle, path):
    source, data, _ = bundle
    data["files"][0]["path"] = path
    source.write_text(json.dumps(data))
    with pytest.raises(artifacts.ArtifactError):
        artifacts.read_manifest(source)


@pytest.mark.parametrize("mutation", ["floating_revision", "wrong_family", "foreign_url", "duplicate", "missing_tokenizer"])
def test_pinning_and_required_model_identity_fail_closed(bundle, mutation):
    source, data, _ = bundle
    if mutation == "floating_revision":
        data["revision"] = "main"
    elif mutation == "wrong_family":
        data["model_id"] = "Qwen/CustomVoice"
    elif mutation == "foreign_url":
        data["files"][0]["url"] = "https://example.com/weights"
    elif mutation == "duplicate":
        data["files"].append(copy.deepcopy(data["files"][0]))
    else:
        data["files"] = data["files"][:2]
    source.write_text(json.dumps(data))
    with pytest.raises(artifacts.ArtifactError):
        artifacts.read_manifest(source)


def test_unexpected_files_and_symlinked_weights_cannot_enter_serving_snapshot(bundle, tmp_path):
    _, manifest, _ = bundle
    cache = tmp_path / "cache"
    root = artifacts.provision(cache, manifest, fetch=fetcher(bundle))
    extra = root / "model.py"
    extra.write_text("not trusted")
    with pytest.raises(artifacts.ArtifactError, match="unexpected"):
        artifacts.verify_snapshot(cache, manifest)
    extra.unlink()
    target = root / "config.json"
    target.unlink()
    target.symlink_to(tmp_path / "outside")
    with pytest.raises(artifacts.ArtifactError):
        artifacts.verify_snapshot(cache, manifest)


def test_offline_or_missing_snapshot_does_not_download(bundle, tmp_path, monkeypatch):
    _, manifest, _ = bundle
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(artifacts.ArtifactError, match="absent"):
        artifacts.verify_snapshot(tmp_path / "cache", manifest)
    with pytest.raises(artifacts.ArtifactError, match="offline"):
        artifacts.provision(tmp_path / "cache", manifest)
