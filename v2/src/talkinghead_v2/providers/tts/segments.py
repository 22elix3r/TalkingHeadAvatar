"""Real multi-segment/one-EOF probe; text and audio are private unless exported deliberately."""

import asyncio
import hashlib
import json
import struct

from talkinghead_v2.contracts import AudioChunk, AudioEOF, Envelope, GenerationKey, TextSegment
from talkinghead_v2.providers.tts.native import QwenNativeProvider, load_native_config


async def run(args):
    texts = json.loads(args.texts.read_text())
    if not isinstance(texts, list) or not 1 <= len(texts) <= 256:
        raise ValueError("segment probe requires 1 to 256 text segments")
    ids = set()
    for item in texts:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip()
            or item["id"] in ids or not isinstance(item.get("text"), str) or not item["text"].strip()
            or len(item["text"].encode()) > 65536):
            raise ValueError("segment probe requires unique IDs and bounded nonempty text")
        ids.add(item["id"])
    config = load_native_config(args.runtime)
    provider = QwenNativeProvider(config, diagnostic_log=args.log)
    key = GenerationKey("segment-qualification", "one-turn", 0)
    samples, spans, produced, eof_count = [], [], [], 0
    async def consume():
        nonlocal eof_count
        async for event in provider.outputs():
            if isinstance(event, AudioChunk):
                if event.start_sample != len(samples):
                    raise RuntimeError("segment boundary has a sample gap/overlap")
                spans.append({"segment_id": event.segment_id, "start_sample": event.start_sample,
                              "end_sample_exclusive": event.end_sample})
                samples.extend(event.samples)
            else:
                eof_count += 1
                if event.end_sample != len(samples):
                    raise RuntimeError("multi-segment EOF cursor mismatch")
    try:
        await provider.open_session(key)
        consumer = asyncio.create_task(consume())
        try:
            for sequence, item in enumerate(texts):
                if not await provider.push(TextSegment(Envelope(key, sequence, "segment-probe"), item["id"],
                    item["text"], "en", config.voice_id, sequence == len(texts)-1)):
                    raise RuntimeError("segment was cancelled")
                produced.append({"id": item["id"], "samples": provider.last_metrics["samples"],
                                 "pcm_sha256": provider.last_metrics["pcm_sha256"]})
            await consumer
        finally:
            if not consumer.done():
                consumer.cancel(); await asyncio.gather(consumer, return_exceptions=True)
    finally:
        await provider.close()
    if eof_count != 1 or len(samples) != sum(row["samples"] for row in produced):
        raise RuntimeError("multi-segment conservation/EOF failure")
    for row in produced:
        if sum(span["end_sample_exclusive"]-span["start_sample"] for span in spans if span["segment_id"] == row["id"]) != row["samples"]:
            raise RuntimeError("segment label conservation failure")
    import soundfile as sf
    args.wave.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.wave, samples, 24000, subtype="FLOAT")
    report = {"schema_version": 1, "evidence_category": "local_observation", "voice_kind": config.voice_kind,
              "model_revision": config.model_revision, "worker_environment": dict(config.environment),
              "text_fixture_sha256": hashlib.sha256(args.texts.read_bytes()).hexdigest(), "sample_rate": 24000,
              "samples": len(samples), "native_segments": produced, "sample_spans": spans, "eof_count": eof_count,
              "pcm_sha256": hashlib.sha256(struct.pack(f"<{len(samples)}f", *samples)).hexdigest(),
              "status": "passed", "human_listening": "pending", "human_cloning_qualified": False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    return report
