"""Real Qwen callback qualification. Production horizons are separate from presentation."""

import asyncio
import hashlib
import json
import math
from pathlib import Path
import platform
import signal
import struct
import time

from talkinghead_v2.contracts import AudioChunk, AudioEOF, Envelope, GenerationKey, TextSegment
from talkinghead_v2.providers.tts.native import QwenNativeProvider, load_native_config


def summarize(metrics: dict) -> dict:
    callbacks = metrics["callbacks"]
    if not callbacks or not metrics["samples"]:
        raise RuntimeError("request produced no real callbacks")
    horizons = {}
    for ms in (100, 300, 500):
        horizons[str(ms)] = next((cb["received_ns"] for cb in callbacks if cb["cumulative_samples"] >= ms*24), None)
    gaps = [b["native_elapsed_ns"]-a["native_elapsed_ns"] for a, b in zip(callbacks, callbacks[1:])]
    return {"sample_rate": 24000, "samples": metrics["samples"], "pcm_sha256": metrics["pcm_sha256"],
            "ttfa_ns": callbacks[0]["received_ns"], "horizons_ns": horizons,
            "max_native_production_gap_ns": max(gaps, default=0), "wall_ns": metrics["wall_ns"],
            "native_end_ns": metrics["native_end_ns"],
            "rtf": metrics["wall_ns"]/1e9/(metrics["samples"]/24000),
            "callback_count": len(callbacks),
            "early_pcm": callbacks[0]["native_elapsed_ns"] + 100_000_000 < metrics["native_end_ns"],
            "callbacks": callbacks}


async def _memory(stop: asyncio.Event, values: list[int]):
    while not stop.is_set():
        try:
            process = await asyncio.create_subprocess_exec("nvidia-smi", "--query-gpu=memory.used",
                "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            output, _ = await asyncio.wait_for(process.communicate(), 2)
            if process.returncode == 0:
                values.append(int(output.decode().splitlines()[0].strip())*1024**2)
        except (OSError, ValueError, TimeoutError, IndexError):
            pass
        try:
            await asyncio.wait_for(stop.wait(), 0.25)
        except TimeoutError:
            pass


async def _one(provider, key, request, save: Path | None, *, already_open=False):
    if not already_open:
        await provider.open_session(key)
    total, digest, chunks, eof = 0, hashlib.sha256(), [], False
    async def consume():
        nonlocal total, eof
        async for event in provider.outputs():
            if isinstance(event, AudioChunk):
                if event.start_sample != total:
                    raise RuntimeError("callback sample gap/overlap")
                payload = struct.pack(f"<{event.sample_count}f", *event.samples)
                digest.update(payload)
                total += event.sample_count
                if save is not None:
                    chunks.extend(event.samples)
            else:
                if event.end_sample != total:
                    raise RuntimeError("native EOF cursor mismatch")
                eof = True
    consumer = asyncio.create_task(consume())
    try:
        segment = TextSegment(Envelope(key, 0, "qualification"), request["id"], request["text"],
                              "en", provider.config.voice_id, True)
        if not await provider.push(segment):
            raise RuntimeError("request cancelled unexpectedly")
        await consumer
        if not eof or total != provider.last_metrics["samples"] or digest.hexdigest() != provider.last_metrics["pcm_sha256"]:
            raise RuntimeError("consumed PCM/EOF differs from actual native callback bytes")
        result = summarize(provider.last_metrics)
        result["id"] = request["id"]
        result["text_sha256"] = hashlib.sha256(request["text"].encode()).hexdigest()
        result["epoch"] = key.epoch
        if save is not None:
            # Optional isolated inference dependency; core tests/imports do not require it.
            import soundfile as sf
            save.parent.mkdir(parents=True, exist_ok=True)
            sf.write(save, chunks, 24000, subtype="FLOAT")
        return result
    finally:
        if not consumer.done():
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)


async def run(args) -> dict:
    config = load_native_config(args.runtime)
    provider = QwenNativeProvider(config, diagnostic_log=args.log)
    adapter_digest = hashlib.sha256(Path(__file__).with_name("native.py").read_bytes()).hexdigest()
    texts = json.loads(args.texts.read_text())
    if not texts:
        raise ValueError("qualification fixture has no requests")
    results, failures, memory = [], [], []
    stop = asyncio.Event()
    monitor = asyncio.create_task(_memory(stop, memory))
    started = None
    epoch = 0
    interrupted = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, interrupted.set)
    try:
        await provider.open_session(GenerationKey("qwen-qualification", "request-0", 0))
        started = time.perf_counter_ns()
        # A whole cohort is mandatory even when the requested soak duration is zero.
        while epoch < len(texts) or time.perf_counter_ns()-started < args.seconds*1e9:
            request = texts[epoch % len(texts)]
            key = GenerationKey("qwen-qualification", f"request-{epoch}", epoch)
            save = args.save_waves / f"native-{epoch:03}.wav" if args.save_waves is not None and epoch < len(texts) else None
            try:
                result = await _one(provider, key, request, save, already_open=epoch == 0)
                results.append(result)
            except Exception as exc:
                failures.append({"id": request["id"], "epoch": epoch, "error": str(exc)})
                await provider.close()
            epoch += 1
            progress = {"elapsed_seconds": (time.perf_counter_ns()-started)/1e9, "completed": len(results),
                        "failures": failures, "last_result": results[-1] if results else None}
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.with_suffix(".progress.json").write_text(json.dumps(progress, indent=2)+"\n")
            with args.report.with_suffix(".events.jsonl").open("a") as trace:
                trace.write(json.dumps({"epoch": epoch-1, "result": results[-1] if results and results[-1]["epoch"] == epoch-1 else None,
                                       "failure": failures[-1] if failures and failures[-1]["epoch"] == epoch-1 else None})+"\n")
            if len(failures) >= 3 or interrupted.is_set():
                break
        elapsed = time.perf_counter_ns()-started
    finally:
        await provider.close()
        stop.set()
        await monitor
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(signum)
    def quantile(values, fraction):
        return sorted(values)[max(0, math.ceil(len(values)*fraction)-1)] if values else None
    report = {"schema_version": 1, "phase": "P02", "evidence_category": "local_observation",
        "boundary": "complete_text_segment_to_native_pcm_callbacks", "actual_device_playback": False,
        "incremental_text_input": False, "incremental_audio_output": True, "voice_kind": config.voice_kind,
        "human_cloning_qualified": False, "human_listening": "pending", "wer": None,
        "model_revision": config.model_revision, "runtime_revision": config.runtime_revision,
        "worker_environment": dict(config.environment),
        "runtime_descriptor_sha256": hashlib.sha256(args.runtime.read_bytes()).hexdigest(),
        "adapter_source_sha256": adapter_digest,
        "text_fixture_sha256": hashlib.sha256(args.texts.read_bytes()).hexdigest(),
        "python": platform.python_version(), "kernel": platform.release(), "model_load_ns": provider.load_ns,
        "requested_duration_seconds": args.seconds, "elapsed_ns": elapsed, "completed_requests": len(results),
        "interrupted": interrupted.is_set(),
        "duration_boundary": "after model/voice readiness through final completed request; excludes shutdown",
        "failed_requests": failures, "requests": results, "stale_native_packets": provider.stale_packets,
        "queue_peak_samples": provider.queue.peak_samples, "queue_max_samples": provider.queue.max_samples,
        "memory": {"device_used_sampled_peak_bytes": max(memory, default=None), "sampling_period_seconds": 0.25,
                   "scope": "whole GPU including desktop; nvidia-smi samples; allocated/reserved not PyTorch metrics",
                   "samples": len(memory)},
        "empirical_nearest_rank": {"rtf_p50": quantile([r["rtf"] for r in results], .5),
            "rtf_p95": quantile([r["rtf"] for r in results], .95),
            "horizon_300ms_p95_ns": quantile([r["horizons_ns"]["300"] for r in results if r["horizons_ns"]["300"] is not None], .95)},
        "functional_streaming_passed": bool(results) and not failures and all(r["early_pcm"] for r in results if r["id"] == "long"),
        "stability_30_minutes_passed": args.seconds >= 1800 and elapsed >= 1800*1e9 and not failures and not interrupted.is_set(),
        "limitations": ["Synthetic reference; no independent listening/WER or human clone qualification",
                        "PCM production horizons are not device presentation or conversation latency",
                        "Native allocated/reserved memory is not exposed; device usage is sampled"]}
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    return report
