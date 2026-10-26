"""Real output-monitor calibration. Captures the selected output monitor, never a microphone."""

import asyncio
from dataclasses import asdict
from fractions import Fraction
import hashlib
import math
import struct
import time

from talkinghead_v2.audio.playback import PortAudioSink
from talkinghead_v2.audio.queue import EpochCancelled
from talkinghead_v2.contracts import AudioChunk, AudioEOF, Envelope
from talkinghead_v2.session.epochs import GenerationSession


async def consume_device(session, key):
    while True:
        try:
            event = await session.queue.get(key)
        except EpochCancelled:
            return False
        if isinstance(event, AudioChunk):
            if not await session.sink.wait_writable(key, event.sample_count):
                return False
        if not session.sink.submit(event):
            return False
        if isinstance(event, AudioEOF):
            return await session.sink.drain(key)


def tone_analysis(payload: bytes) -> dict:
    import numpy as np
    audio = np.frombuffer(payload[:len(payload)//4*4], dtype="<f4")
    window = 1200  # 25 ms: integer cycles at both 440 and 880 Hz.
    parts = audio[:len(audio)//window*window].reshape(-1, window)
    amplitudes = {}
    t = np.arange(window)/48000
    for frequency in (440, 880):
        a = parts @ np.sin(2*np.pi*frequency*t)
        b = parts @ np.cos(2*np.pi*frequency*t)
        amplitudes[frequency] = 2*np.sqrt(a*a+b*b)/window
    old = np.flatnonzero(amplitudes[440] > .01)
    new = np.flatnonzero(amplitudes[880] > .01)
    after = amplitudes[440][new[0]+2:] if new.size else np.array([])
    reappeared = bool(after.size and np.max(after) > .01)
    return {"old_tone_detected": bool(old.size), "new_tone_detected": bool(new.size),
            "old_tone_reappeared_after_new_plus_50ms": reappeared,
            "old_tone_max_after_resume": float(np.max(after)) if after.size else None,
            "old_tone_peak": float(np.max(amplitudes[440])) if parts.size else 0,
            "new_tone_peak": float(np.max(amplitudes[880])) if parts.size else 0,
            "transition_gap_ms": max(0, int(new[0]-old[-1]-1)*25) if old.size and new.size else None,
            "window_ms": 25, "threshold": .01, "samples": int(audio.size),
            "passed": bool(old.size and new.size and not reappeared)}


async def run(args) -> dict:
    process = await asyncio.create_subprocess_exec("pactl", "get-default-sink", stdout=asyncio.subprocess.PIPE)
    stdout, _ = await process.communicate()
    if process.returncode:
        raise RuntimeError("cannot identify default output monitor")
    monitor = stdout.decode().strip()+".monitor"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    capture = await asyncio.create_subprocess_exec("parec", "--device="+monitor, "--format=float32le",
        "--rate=48000", "--channels=1", "--latency-msec=20", stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    recorded = bytearray()
    async def collect():
        while block := await capture.stdout.read(32768):
            recorded.extend(block)
            if len(recorded) > 10*48000*4:
                raise RuntimeError("calibration capture exceeded ten-second bound")
    recording = asyncio.create_task(collect())
    sink = PortAudioSink(device=args.device)
    session = GenerationSession("device-calibration", sink, max_queue_duration=Fraction(1, 2))
    async def feed(key, frequency, count):
        cursor, sequence = 0, 0
        while cursor < count:
            take = min(960, count-cursor)
            values = tuple(.12*math.sin(2*math.pi*frequency*i/48000) for i in range(cursor, cursor+take))
            if not await session.push(AudioChunk(Envelope(key, sequence, "tone-calibration"), "tone", 48000, cursor, values)):
                return False
            cursor += take; sequence += 1
        return await session.finish(AudioEOF(Envelope(key, sequence, "tone-eof"), 48000, count))
    tasks = []
    try:
        await asyncio.sleep(.2)  # Capture pre-roll, never PCM pacing.
        await sink.open_device()
        first = session.start("old-tone", 48000)
        tasks = [asyncio.create_task(feed(first.key, 440, 96000)), asyncio.create_task(consume_device(session, first.key))]
        async def first_dac():
            while sink.first_dac_ns is None:
                await asyncio.sleep(.005)
        await asyncio.wait_for(first_dac(), 2)
        await asyncio.sleep(max(0, (sink.first_dac_ns+250_000_000-time.perf_counter_ns())/1e9))
        cancelled_ns = time.perf_counter_ns()
        software = session.cancel("calibration interrupt")
        closed = await sink.abort_and_close()
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        old_report = sink.report()
        await sink.open_device()
        second = session.start("new-tone", 48000)
        tasks = [asyncio.create_task(feed(second.key, 880, 28800)), asyncio.create_task(consume_device(session, second.key))]
        result = await asyncio.wait_for(asyncio.gather(*tasks), 3)
        if result != [True, True]:
            raise RuntimeError("new tone did not reach explicit EOF/device drain")
        new_report = sink.report()
        await sink.abort_and_close()
        await asyncio.sleep(.2)
    finally:
        await sink.abort_and_close()
        for task in tasks:
            if not task.done(): task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if capture.returncode is None:
            capture.terminate()
        await capture.wait()
        await recording
    args.capture.parent.mkdir(parents=True, exist_ok=True)
    args.capture.write_bytes(recorded)
    analysis = tone_analysis(recorded)
    return {"schema_version": 1, "evidence_category": "local_observation", "boundary": "PortAudio to output-monitor loopback",
            "microphone_captured": False, "monitor": monitor, "actual_device_stream": True,
            "cancel_requested_monotonic_ns": cancelled_ns, "software_ack": asdict(software), "device_close_ack": closed,
            "old": old_report, "resumed": new_report, "capture_sha256": hashlib.sha256(recorded).hexdigest(),
            "analysis": analysis, "passed": analysis["passed"],
            "limitations": ["Output monitor is not an acoustic recording or human listening",
                            "No exact monitor-to-cancel clock mapping; no wall-clock stale-audio bound claimed",
                            "Device abort/close plus new stream qualified only on this local output profile"]}
