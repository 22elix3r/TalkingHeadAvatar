"""One preloaded Qwen CUDA/F16 worker; complete text per segment, real PCM callbacks."""

import asyncio
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time

from talkinghead_v2.assets.artifacts import ArtifactError, MODEL_ID, read_manifest, sha256_file, verify_snapshot
from talkinghead_v2.assets.voices import read_voice
from talkinghead_v2.audio.queue import EpochCancelled, MediaQueue
from talkinghead_v2.contracts import AudioChunk, AudioEOF, ContractError, Envelope, GenerationKey, ProviderCapabilities, TextSegment

HEADER = struct.Struct("<cIIQQ")


class TtsError(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeConfig:
    command: tuple[str, ...]
    model_revision: str
    runtime_revision: str
    tenant_id: str
    voice_id: str
    voice_kind: str
    environment: tuple[tuple[str, str | None], ...] = ()


def load_native_config(path: Path) -> NativeConfig:
    """Operator-owned build descriptor. Verify all files before executing the local worker."""
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or data.get("backend") != "qwen_cpp_cuda_f16":
        raise ArtifactError("unsupported native runtime descriptor")
    def local(name):
        return (path.parent / data[name]).resolve()
    manifest = read_manifest(local("manifest"))
    verify_snapshot(local("cache"), manifest)
    voice = read_voice(local("voice_registry"), tenant_id=data["tenant_id"])
    worker, models, embedding = local("worker"), local("models"), local("embedding")
    if worker.is_symlink() or sha256_file(worker) != data["worker_sha256"]:
        raise ArtifactError("native worker identity differs from the recorded local build")
    expected = data["derived_models"]
    names = {"qwen3-tts-1.7b-base-f16.gguf", "qwen3-tts-tokenizer-f16.gguf"}
    if set(expected) != names or {p.name for p in models.iterdir()} != names:
        raise ArtifactError("native model directory must contain exactly the selected Base/tokenizer pair")
    for name in names:
        if (models/name).is_symlink() or sha256_file(models/name) != expected[name]:
            raise ArtifactError("derived model identity mismatch")
    if embedding.is_symlink() or sha256_file(embedding) != data["embedding_sha256"]:
        raise ArtifactError("speaker embedding identity mismatch")
    values = json.loads(embedding.read_text())
    if not isinstance(values, list) or len(values) != data["embedding_dimension"] or not all(
        type(x) in (int, float) and math.isfinite(x) for x in values
    ):
        raise ArtifactError("invalid speaker embedding")
    identity = data["reference_identity"]
    if (identity["reference_sha256"] != voice["audio_sha256"] or
        identity["model_revision"] != manifest["revision"] or identity["conditioning"] != "x-vector" or
        identity["tenant_id"] != data["tenant_id"] or identity["voice_id"] != voice["voice_id"]):
        raise ArtifactError("embedding reference/model/tenant/conditioning binding mismatch")
    disable_graphs, async_decode = data.get("disable_cuda_graphs", True), data.get("async_decode", False)
    if type(disable_graphs) is not bool or type(async_decode) is not bool:
        raise ArtifactError("runtime scheduling choices must be explicit booleans")
    environment = (("GGML_CUDA_DISABLE_GRAPHS", "1" if disable_graphs else None),
                   ("THA_QWEN_ASYNC_DECODE", "1" if async_decode else "0"))
    return NativeConfig((str(worker), str(models), str(embedding)), manifest["revision"],
                        data["runtime_revision"], data["tenant_id"], voice["voice_id"], voice["kind"], environment)


@dataclass
class _Request:
    serial: int
    key: GenerationKey
    segment_id: str
    started_ns: int
    done: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False
    compute_stopped: bool = False
    failure: str | None = None
    elapsed_ns: int = 0
    native_end_ns: int = 0
    callbacks: list[dict] = field(default_factory=list)
    sample_count: int = 0
    digest: object = field(default_factory=hashlib.sha256)


class QwenNativeProvider:
    """Single-loop, one-voice provider. Await push while consuming outputs concurrently.

    push backpressures through actual synthesis; it never buffers a whole reply's text.
    cancel immediately fences PCM, then acknowledges actual compute stop or worker exit.
    """

    def __init__(self, config: NativeConfig, *, diagnostic_log: Path, generation_timeout: float = 180):
        self.config = config
        self.diagnostic_log = diagnostic_log
        if generation_timeout <= 0:
            raise ContractError("generation timeout must be positive")
        self.generation_timeout = generation_timeout
        self.capabilities = ProviderCapabilities(MODEL_ID, config.model_revision, config.runtime_revision,
            (24000,), False, True, "cooperative", 24000, 4*1920, 6*1920, 2400)
        self.queue = MediaQueue(24000, Fraction(1), 16)
        self._process = None
        self._reader_task = None
        self._ready = None
        self._key = None
        self._request = None
        self._serial = 0
        self._cursor = 0
        self._sequence = 0
        self._text_sequence = -1
        self._segments = set()
        self._lock = asyncio.Lock()
        self._flushed = False
        self._eof_consumed = False
        self._failure = None
        self.last_metrics = None
        self.load_ns = 0
        self.stale_packets = 0

    async def _send(self, kind: bytes, payload: bytes = b""):
        if self._process is None or self._process.returncode is not None:
            raise TtsError("native worker is not running")
        self._process.stdin.write(kind + struct.pack("<I", len(payload)) + payload)
        await self._process.stdin.drain()

    async def open_session(self, key: GenerationKey):
        if self._key is not None and (not self._flushed or not self._eof_consumed or self._lock.locked()):
            raise ContractError("flush and consume EOF, or cancel, before provider restart")
        if key.session_id != self.queue._session_id and self.queue._session_id is not None:
            raise ContractError("one provider instance belongs to one session")
        if self._process is None:
            self.diagnostic_log.parent.mkdir(parents=True, exist_ok=True)
            self._ready = asyncio.get_running_loop().create_future()
            environment = dict(os.environ)
            for name, value in self.config.environment:
                if value is None:
                    environment.pop(name, None)
                else:
                    environment[name] = value
            with self.diagnostic_log.open("ab") as log:
                self._process = await asyncio.create_subprocess_exec(*self.config.command,
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log,
                    limit=256*1024, env=environment)
            self._reader_task = asyncio.create_task(self._read())
            try:
                await asyncio.wait_for(self._ready, 60)
            except BaseException:
                await self._stop()
                raise
        if self._failure:
            raise TtsError(self._failure)
        self.queue.reset(key)
        self._key = key
        self._cursor = self._sequence = 0
        self._text_sequence = -1
        self._segments = set()
        self._flushed = False
        self._eof_consumed = False

    async def _read(self):
        try:
            while True:
                kind, rate, count, elapsed_ns, serial = HEADER.unpack(await self._process.stdout.readexactly(HEADER.size))
                if kind == b"A":
                    if rate != 24000 or count > 48000:
                        raise TtsError("native callback has invalid rate/size")
                    payload = await self._process.stdout.readexactly(count*4)
                elif kind == b"E":
                    if count > 65536:
                        raise TtsError("oversized worker error")
                    payload = await self._process.stdout.readexactly(count)
                elif kind in (b"R", b"D", b"C") and count == 0:
                    payload = b""
                else:
                    raise TtsError(f"invalid native protocol frame kind={kind.hex()} rate={rate} count={count} request={serial}")
                if kind == b"R":
                    if rate != 24000 or self._ready.done():
                        raise TtsError("invalid/duplicate worker readiness")
                    self.load_ns = elapsed_ns
                    self._ready.set_result(True)
                    continue
                if kind == b"E" and not self._ready.done():
                    raise TtsError(payload.decode("utf-8", errors="replace"))
                request = self._request
                if request is None or request.serial != serial:
                    self.stale_packets += 1
                    continue
                if request.done.is_set():
                    self.stale_packets += 1
                    continue
                if kind == b"A":
                    request.sample_count += count
                    request.digest.update(payload)
                    request.callbacks.append({"samples": count, "native_elapsed_ns": elapsed_ns,
                        "received_ns": time.perf_counter_ns()-request.started_ns,
                        "cumulative_samples": request.sample_count})
                    if request.cancelled or self._key != request.key:
                        self.stale_packets += 1
                        continue
                    samples = struct.unpack(f"<{count}f", payload)
                    for offset in range(0, count, 2400):
                        chunk = AudioChunk(Envelope(request.key, self._sequence, f"native-{serial}"),
                            request.segment_id, 24000, self._cursor, samples[offset:offset+2400])
                        if not await self.queue.put(chunk):
                            break
                        self._cursor = chunk.end_sample
                        self._sequence += 1
                else:
                    request.elapsed_ns = time.perf_counter_ns()-request.started_ns
                    request.native_end_ns = elapsed_ns
                    if kind == b"E":
                        request.failure = payload.decode("utf-8", errors="replace")
                    elif kind == b"C":
                        request.cancelled = True
                        request.compute_stopped = True
                    elif kind == b"D":
                        request.compute_stopped = True
                    request.done.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failure = str(exc) or type(exc).__name__
            if self._ready is not None and not self._ready.done():
                self._ready.set_exception(TtsError(self._failure))
            if self._request is not None:
                self._request.failure = self._failure
                self._request.done.set()
            self.queue.reset(None)
            if self._process is not None and self._process.returncode is None:
                self._process.terminate()

    async def push(self, segment: TextSegment) -> bool:
        async with self._lock:
            if self._key != segment.envelope.key:
                return False
            if (self._flushed or segment.envelope.sequence_no <= self._text_sequence or
                    segment.segment_id in self._segments or len(self._segments) >= 256):
                raise ContractError("closed, duplicate, reordered or excessive text segments")
            if segment.voice_id != self.config.voice_id or segment.language != "en":
                raise ContractError("voice/language mismatch; no silent substitution")
            payload = segment.text.encode("utf-8")
            if len(payload) > 65528:
                raise ContractError("text segment exceeds the native request bound")
            self._text_sequence = segment.envelope.sequence_no
            self._segments.add(segment.segment_id)
            self._serial += 1
            request = _Request(self._serial, self._key, segment.segment_id, time.perf_counter_ns())
            self._request = request
            try:
                await self._send(b"G", struct.pack("<Q", request.serial)+payload)
                await asyncio.wait_for(request.done.wait(), self.generation_timeout)
                self.last_metrics = {"serial": request.serial, "samples": request.sample_count,
                    "wall_ns": request.elapsed_ns, "callbacks": request.callbacks,
                    "native_end_ns": request.native_end_ns,
                    "pcm_sha256": request.digest.hexdigest(), "cancelled": request.cancelled,
                    "failure": request.failure}
                if request.cancelled or self._key != request.key:
                    return False
                if request.failure:
                    await self.cancel()
                    raise TtsError(request.failure)
                if segment.is_last:
                    return await self._flush()
                return True
            except asyncio.CancelledError:
                await self.cancel()
                raise
            except Exception:
                if self._key == request.key:
                    request.failure = request.failure or "provider request failed"
                    await self.cancel()
                raise

    async def _flush(self):
        if self._key is None:
            return False
        if self._flushed:
            raise ContractError("duplicate provider flush")
        self._flushed = True
        return await self.queue.put(AudioEOF(Envelope(self._key, self._sequence, "native-eof"), 24000, self._cursor))

    async def flush(self):
        async with self._lock:
            try:
                return await self._flush()
            except asyncio.CancelledError:
                await self.cancel()
                raise

    async def outputs(self):
        key = self._key
        if key is None:
            raise ContractError("open a provider session before consuming audio")
        while True:
            try:
                event = await self.queue.get(key)
            except EpochCancelled:
                if self._failure:
                    raise TtsError(self._failure)
                return
            if isinstance(event, AudioEOF):
                self._eof_consumed = True
            yield event
            if isinstance(event, AudioEOF):
                return

    async def cancel(self, *, timeout: float = 2.0):
        start = time.perf_counter_ns()
        self._key = None
        discarded, items = self.queue.reset(None)
        request = self._request
        recovery = "cooperative"
        if request is not None and (request.failure or self._failure):
            recovery = "worker_terminated_after_failure"
            await self._stop()
        elif request is not None and not request.compute_stopped:
            request.cancelled = True
            await self._send(b"C")
            try:
                await asyncio.wait_for(request.done.wait(), timeout)
            except TimeoutError:
                recovery = "worker_terminated_cold_reload_required"
                await self._stop()
            else:
                if not request.compute_stopped:
                    recovery = "worker_terminated_after_failure"
                    await self._stop()
        return {"compute_stopped": True, "scope": "native_worker_compute_only", "device_flushed": False,
                "recovery": recovery, "elapsed_ns": time.perf_counter_ns()-start,
                "discarded_samples": discarded, "discarded_items": items}

    async def _stop(self):
        process, reader = self._process, self._reader_task
        self._process = self._reader_task = None
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 2)
            except TimeoutError:
                process.kill()
                await process.wait()
        if reader is not None and reader is not asyncio.current_task():
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        if self._request is not None:
            self._request.compute_stopped = True
            self._request.done.set()
        self._failure = None

    async def close(self):
        await self.cancel()
        if self._process is not None and self._process.returncode is None:
            await self._send(b"X")
            try:
                await asyncio.wait_for(self._process.wait(), 5)
            except TimeoutError:
                pass
        await self._stop()
