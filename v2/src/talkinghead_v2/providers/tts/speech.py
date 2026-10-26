"""Real Qwen → P01 ledger/resampler/epoch queue → callback-paced local playback."""

import asyncio
from dataclasses import asdict
import json
import time

from talkinghead_v2.audio.capture import OutputMonitorCapture
from talkinghead_v2.audio.playback import PortAudioSink
from talkinghead_v2.audio.qualify import consume_device
from talkinghead_v2.contracts import AudioChunk, Envelope, TextSegment
from talkinghead_v2.providers.tts.benchmark import summarize
from talkinghead_v2.providers.tts.native import QwenNativeProvider, load_native_config
from talkinghead_v2.session.epochs import GenerationSession


async def run(args):
    config = load_native_config(args.runtime)
    provider = QwenNativeProvider(config, diagnostic_log=args.log)
    sink = PortAudioSink(device=args.device)
    session = GenerationSession("local-qwen-playback", sink)
    capture = OutputMonitorCapture()
    await capture.start()
    turns = []

    async def turn(text, *, interrupted=False):
        await sink.open_device()
        started = session.start(f"turn-{len(turns)}", 24000)
        key = started.key
        await provider.open_session(key)
        captured_start = capture.arm()
        producer_finished = None
        async def produce():
            nonlocal producer_finished
            result = await provider.push(TextSegment(Envelope(key, 0, "local-playback"), "speech",
                text, "en", config.voice_id, True))
            producer_finished = time.perf_counter_ns()
            return result
        async def bridge():
            async for event in provider.outputs():
                accepted = await session.push(event) if isinstance(event, AudioChunk) else await session.finish(event)
                if not accepted:
                    return False
            return True
        tasks = [asyncio.create_task(produce()), asyncio.create_task(bridge()),
                 asyncio.create_task(consume_device(session, key))]
        try:
            if interrupted:
                async def first_dac():
                    while sink.first_dac_ns is None:
                        if tasks[0].done() and tasks[0].exception():
                            await tasks[0]
                        await asyncio.sleep(.005)
                await asyncio.wait_for(first_dac(), 30)
                await asyncio.sleep(max(0, (sink.first_dac_ns+args.interrupt_after*1e9-time.perf_counter_ns())/1e9))
                interrupt_ns = time.perf_counter_ns()
                software = session.cancel("explicit local speech interruption")
                device_ack, compute_ack = await asyncio.gather(sink.abort_and_close(), provider.cancel())
                result = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
                return {"epoch": key.epoch, "interrupted": True, "interrupt_monotonic_ns": interrupt_ns,
                        "software_ack": asdict(software), "device_ack": device_ack, "compute_ack": compute_ack,
                        "task_outcomes": [str(value) for value in result], "device": sink.report(),
                        "capture_first_audio_observed_ns": capture.first_audio_observed_ns}
            result = await asyncio.wait_for(asyncio.gather(*tasks), 180)
            if result != [True, True, True]:
                raise RuntimeError("local speech did not reach provider EOF and device drain")
            observed = capture.first_audio_observed_ns
            metrics = summarize(provider.last_metrics)
            report = {"epoch": key.epoch, "interrupted": False, "capture_armed_ns": captured_start,
                      "capture_first_audio_observed_ns": observed, "producer_finished_monotonic_ns": producer_finished,
                      "output_monitor_audio_observed_before_generation_returned": observed is not None and observed < producer_finished,
                      "native_source_emitted_early_pcm": metrics["early_pcm"], "native": metrics, "device": sink.report(),
                      "actor_queue_peak_samples": session.queue.peak_samples, "actor_queue_max_samples": session.queue.max_samples}
            if not report["output_monitor_audio_observed_before_generation_returned"] or not metrics["early_pcm"]:
                raise RuntimeError("early real streaming/presentation was not established")
            return report
        finally:
            for task in tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await sink.abort_and_close()

    try:
        await asyncio.sleep(.2)  # Output-monitor pre-roll only, never playback pacing.
        turns.append(await turn(args.text, interrupted=args.interrupt_after is not None))
        if args.interrupt_after is not None:
            await asyncio.sleep(.1)  # Conservative quiet resume boundary after old stream close.
            turns.append(await turn(args.resume_text))
        await asyncio.sleep(.1)
    finally:
        await provider.close()
        await sink.abort_and_close()
        capture_report = await capture.close(args.capture)
    report = {"schema_version": 1, "phase": "P02", "evidence_category": "local_observation",
              "boundary": "Qwen native callbacks through P01 to PortAudio and output-monitor observation",
              "actual_device_playback": True, "voice_kind": config.voice_kind, "human_cloning_qualified": False,
              "human_listening": "pending", "turns": turns, "capture": capture_report,
              "receiver_safe_browser_cancel": "not_qualified", "status": "passed",
              "limitations": ["Output-monitor observation is not acoustic capture or human quality",
                              "PortAudio timestamp uncertainty and device profile remain explicit",
                              "Native production RTF must be measured in the separate unpaced callback cohort"]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    return report
