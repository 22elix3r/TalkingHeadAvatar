"""Executable P01 contract fixture; synthetic scheduling, not device presentation."""

import asyncio
from dataclasses import asdict
from fractions import Fraction
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import platform
import struct

from talkinghead_v2 import __version__
from talkinghead_v2.audio.resample import RationalResampler
from talkinghead_v2.contracts import AudioChunk, AudioEOF, Envelope, MotionFrame, RigIdentity
from talkinghead_v2.session.epochs import GenerationSession
from talkinghead_v2.session.synthetic import SyntheticSink


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(f"P01 replay failed: {detail}")


def pcm_hash(samples: tuple[float, ...]) -> str:
    """Digest explicit IEEE float32 little-endian samples, never Python object repr."""
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(struct.pack("<f", sample))
    return digest.hexdigest()


def procedural_pcm(count: int, silence_start: int = 0, silence_end: int = 0) -> tuple[float, ...]:
    # Fixed integer ramp plus impulses, with an explicit silence interval. No PRNG state.
    return tuple(0.0 if silence_start <= i < silence_end else
                 (8192 if i % 101 == 0 else ((i*73) % 4096)-2048) / 32768
                 for i in range(count))


def _source_digest() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


async def _replay() -> dict[str, object]:
    raw = files("talkinghead_v2").joinpath("fixtures/p01.json").read_bytes()
    fixture = json.loads(raw)
    output_rate = fixture["output_rate"]
    sink = SyntheticSink(output_rate)
    session = GenerationSession("synthetic-session", sink,
        max_queue_duration=Fraction(fixture["queue_max_ms"], 1000),
        max_queue_items=fixture["queue_max_items"], output_chunk_samples=fixture["output_chunk_samples"])
    rig = RigIdentity("procedural-rig-v1", "sha256:" + "0"*64, "sha256:" + "1"*64, ("mouth",))

    async def feed(key, spec, samples, *, eof=True):
        cursor = 0
        for sequence, count in enumerate(spec["chunks"]):
            event = AudioChunk(Envelope(key, sequence, "synthetic-trace"), "fixture-segment",
                               spec["native_rate"], cursor, samples[cursor:cursor+count])
            if not await session.push(event):
                return False
            cursor += count
        return await session.finish(AudioEOF(Envelope(key, len(spec["chunks"]), "synthetic-trace"),
                                             spec["native_rate"], cursor)) if eof else True

    cancelled_spec = fixture["cancelled"]
    cancelled_pcm = procedural_pcm(sum(cancelled_spec["chunks"]))
    first = session.start("cancelled-turn", cancelled_spec["native_rate"], rig=rig)
    producer = asyncio.create_task(feed(first.key, cancelled_spec, cancelled_pcm))
    try:
        await asyncio.wait_for(session.queue.put_blocked.wait(), timeout=5)
        _require(sink.submit(session.queue.get_nowait(first.key)), "initial PCM rejected")
        held_callback = session.queue.get_nowait(first.key)
        ack = session.cancel("fixture interruption while producer is backpressured")
        _require(not await asyncio.wait_for(producer, timeout=5), "cancelled producer continued")
    finally:
        if not producer.done():
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
    _require(ack.discarded_queue_items > 0, "cancel did not clear queued PCM")
    _require(not sink.submit(held_callback), "dequeued old callback reached sink")
    late_audio = AudioChunk(Envelope(first.key, 999, "late-trace"), "late", first.native_rate, 0, (0.2,))
    _require(not await session.push(late_audio), "late audio accepted")
    _require(not await session.finish(AudioEOF(Envelope(first.key, 1000, "late-trace"), first.native_rate, 0)),
             "late EOF accepted")
    late_motion = MotionFrame(Envelope(first.key, 0, "late-motion"), rig, 0, output_rate, 480, (0.0,))
    _require(not sink.submit_motion(late_motion), "late motion accepted")

    spec = fixture["completed"]
    pcm = procedural_pcm(sum(spec["chunks"]), spec["silence_start"], spec["silence_end"])
    second = session.start("completed-turn", spec["native_rate"], rig=rig)
    _require(not sink.submit(held_callback), "old PCM accepted after restart")
    _require(not sink.submit_motion(late_motion), "old motion accepted after restart")
    current_motion = MotionFrame(Envelope(second.key, 0, "current-motion"), rig, 0, output_rate, 480, (0.0,))
    _require(sink.submit_motion(current_motion), "current motion rejected")

    async def consume():
        while True:
            event = await session.queue.get(second.key)
            _require(sink.submit(event), "current PCM/EOF rejected")
            if isinstance(event, AudioEOF):
                return

    consumer = asyncio.create_task(consume())
    try:
        _require(await asyncio.wait_for(feed(second.key, spec, pcm), timeout=5), "restarted producer rejected")
        await asyncio.wait_for(consumer, timeout=5)
    finally:
        if not consumer.done():
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
    actual = tuple(sample for item in sink.records if item.chunk.envelope.key == second.key for sample in item.chunk.samples)
    reference = RationalResampler(spec["native_rate"], output_rate)
    body = reference.push(pcm)
    tail = reference.finish()
    expected = (len(pcm)*output_rate + spec["native_rate"]-1) // spec["native_rate"]
    _require(len(actual) == expected and actual == body+tail, "resampled PCM count/content mismatch")
    _require(len(sink.eofs) == 1 and sink.eofs[0].valid_end_sample == Fraction(len(pcm)*output_rate, spec["native_rate"]),
             "EOF count or exact valid duration mismatch")
    _require(session.queue.samples == 0 and session.queue.item_count == 0, "queue did not drain")
    _require(session.queue.peak_samples <= session.queue.max_samples, "queue duration exceeded")
    _require(session.queue.peak_items <= session.queue.max_items, "queue item limit exceeded")
    for previous, following in zip(sink.records, sink.records[1:]):
        _require(previous.end_ns == following.start_ns, "session PTS gap/overlap")
    lock = Path(__file__).parents[2] / "uv.lock"
    return {
        "schema_version": 1, "phase": "P01", "evidence_category": "local_observation",
        "boundary": "synthetic_in_process_scheduling_only", "actual_device_playback": False,
        "performance_or_quality_claim": False, "package_version": __version__,
        "python": platform.python_version(), "os": platform.system(), "kernel": platform.release(),
        "architecture": platform.machine(), "source_revision": None,
        "source_sha256": _source_digest(),
        "dependency_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "fixture_sha256": hashlib.sha256(raw).hexdigest(), "fixture": fixture,
        "cancelled": {"epoch": first.key.epoch, "input_samples": len(cancelled_pcm),
                      "input_pcm_sha256": pcm_hash(cancelled_pcm), "ack": asdict(ack)},
        "completed": {"epoch": second.key.epoch, "input_samples": len(pcm),
                      "input_pcm_sha256": pcm_hash(pcm), "output_samples": len(actual),
                      "output_pcm_sha256": pcm_hash(actual), "reference_parity": actual == body+tail,
                      "eof_tail_samples": len(tail), "valid_end_sample": [reference.valid_end_sample.numerator,
                                                                                  reference.valid_end_sample.denominator],
                      "last_callback_samples": spec["chunks"][-1], "eof_count": len(sink.eofs)},
        "queue": {"max_samples": session.queue.max_samples, "peak_samples": session.queue.peak_samples,
                  "max_items": session.queue.max_items, "peak_items": session.queue.peak_items,
                  "backpressure_waits": session.queue.blocked_puts},
        "stale_callbacks_rejected": session.rejected_stale + sink.rejected_stale,
        "scheduled_audio_spans": [
            {"key": asdict(item.chunk.envelope.key), "sequence_no": item.chunk.envelope.sequence_no,
             "trace_id": item.chunk.envelope.trace_id, "segment_id": item.chunk.segment_id,
             "start_sample": item.chunk.start_sample, "end_sample_exclusive": item.chunk.end_sample,
             "sample_rate": item.chunk.sample_rate, "session_start_sample": item.session_start_sample,
             "clock_id": "synthetic_audio_schedule", "start_ns": [item.start_ns.numerator, item.start_ns.denominator],
             "end_ns": [item.end_ns.numerator, item.end_ns.denominator]}
            for item in sink.records
        ],
        "scheduled_eof": {"key": asdict(sink.eofs[0].envelope.key),
                          "sequence_no": sink.eofs[0].envelope.sequence_no,
                          "end_sample": sink.eofs[0].end_sample,
                          "valid_end_sample": [sink.eofs[0].valid_end_sample.numerator,
                                               sink.eofs[0].valid_end_sample.denominator]},
        "continuous_session_pts": True, "status": "passed",
        "limitations": ["No device buffers, receiver, neural provider, render or compute-cancel qualification",
                        "Reference FIR is CPU correctness code; real-time suitability is unmeasured",
                        "Uncommitted implementation identified by package source digest, not a Git release"],
    }


def run_replay() -> dict[str, object]:
    return asyncio.run(_replay())
