"""Immutable, hash-verified public model snapshots, independent of inference imports."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from urllib.request import urlopen

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


class ArtifactError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or data.get("model_id") != MODEL_ID:
        raise ArtifactError("only the selected Qwen Base 1.7B manifest schema is supported")
    revision = data.get("revision", "")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ArtifactError("model revision must be a full immutable Git SHA")
    entries = data.get("files")
    if not isinstance(entries, list) or not entries:
        raise ArtifactError("manifest needs a nonempty file inventory")
    seen = set()
    for entry in entries:
        name = entry.get("path", "")
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or ".." in relative.parts or "\\" in name or
                relative.as_posix() != name or name in seen or
                relative.suffix not in {".json", ".txt", ".safetensors", ".md"}):
            raise ArtifactError("unsafe, duplicate or unsupported artifact path")
        seen.add(name)
        if type(entry.get("size")) is not int or not 0 < entry["size"] < 50*1024**3:
            raise ArtifactError("invalid artifact byte size")
        if not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ArtifactError("invalid SHA-256")
        expected = f"https://huggingface.co/{MODEL_ID}/resolve/{revision}/{name}"
        if entry.get("url") != expected:
            raise ArtifactError("artifact URL must bind the selected repo, revision and path")
    required = {"config.json", "model.safetensors", "speech_tokenizer/config.json",
                "speech_tokenizer/model.safetensors", "vocab.json", "merges.txt"}
    if not required <= seen:
        raise ArtifactError("manifest is missing required model/tokenizer files")
    return data


def snapshot_path(cache: Path, manifest: dict) -> Path:
    return cache / "models" / "Qwen3-TTS-12Hz-1.7B-Base" / manifest["revision"]


def _check_file(path: Path, entry: dict) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["size"]:
        raise ArtifactError(f"missing/unsafe/incorrect-size artifact: {entry['path']}")
    if sha256_file(path) != entry["sha256"]:
        raise ArtifactError(f"artifact hash mismatch: {entry['path']}")


def verify_snapshot(cache: Path, manifest: dict) -> Path:
    """Read only. Absence/corruption fails; this function never repairs or downloads."""
    root = snapshot_path(cache, manifest)
    if root.is_symlink() or not root.is_dir():
        raise ArtifactError("verified snapshot absent; run explicit artifacts provision first")
    for entry in manifest["files"]:
        target = root / entry["path"]
        if any(parent.is_symlink() for parent in target.parents if parent != root.parent):
            raise ArtifactError("symlink in artifact path")
        _check_file(target, entry)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != {entry["path"] for entry in manifest["files"]}:
        raise ArtifactError("snapshot has unexpected files; no remote code or extra weights permitted")
    return root


@contextmanager
def _lock(cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".provision.lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def provision(cache: Path, manifest: dict, *, fetch=None) -> Path:
    """Deliberate network operation, size/hash checked before immutable publication."""
    if fetch is None:
        def fetch(url):
            if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
                raise ArtifactError("offline mode forbids fetching; startup must use artifacts verify")
            return urlopen(url, timeout=60)
    cache = cache.resolve()
    with _lock(cache):
        destination = snapshot_path(cache, manifest)
        if destination.exists():
            return verify_snapshot(cache, manifest)
        blobs = cache / "blobs"
        blobs.mkdir(exist_ok=True)
        required = sum(e["size"] for e in manifest["files"] if not (blobs / e["sha256"]).exists())
        if shutil.disk_usage(cache).free < required + 1024**3:
            raise ArtifactError("insufficient disk space for artifacts plus 1 GiB reserve")
        for entry in manifest["files"]:
            blob = blobs / entry["sha256"]
            if blob.exists():
                _check_file(blob, entry)
                continue
            temporary = blobs / (entry["sha256"] + ".part")
            try:
                digest, count = hashlib.sha256(), 0
                with fetch(entry["url"]) as source, temporary.open("xb") as output:
                    while block := source.read(min(1024*1024, entry["size"]-count+1)):
                        count += len(block)
                        if count > entry["size"]:
                            raise ArtifactError(f"download exceeds declared size: {entry['path']}")
                        digest.update(block)
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                if count != entry["size"] or digest.hexdigest() != entry["sha256"]:
                    raise ArtifactError(f"download size/hash mismatch: {entry['path']}")
                temporary.chmod(0o444)
                temporary.rename(blob)
            finally:
                temporary.unlink(missing_ok=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=destination.parent))
        try:
            for entry in manifest["files"]:
                target = staging / entry["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(blobs / entry["sha256"], target)
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return verify_snapshot(cache, manifest)
