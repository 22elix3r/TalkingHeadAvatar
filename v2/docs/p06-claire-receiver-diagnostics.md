# Claire receiver frame diagnostics

Local observations, 2026-10-28. The live-Qwen/Claire qualification now records received source-frame IDs and callback-arrival offsets, the checked receiver queue's high-water marks, and a sanitized snapshot of the SDK's inbound video RTP counters. It exports no track, transport, candidate or session identifiers from that snapshot. Stats failures are reported as unavailable; they do not become zero-loss observations. [Aggregated evidence](../benchmarks/reports/p06-claire-receiver-diagnostics.json) retains seven same-utterance runs and source/binary identities.

The earlier integration run observed **312/324** source frames. Three uninstrumented repeats each observed **324/324**. Three subsequent instrumented runs observed **322/324, 322/324 and 323/324**, with no source-video skips reported by the sender. Every run received the first and final source frames. In the instrumented runs, the checked video queue peaked at one, two and one frames out of its 30-frame bound. The SDK snapshots each reported zero packets lost or discarded, zero NACK/PLI/FIR and zero `frames_dropped`; they reported 352 frames received and 350 decoded. These are asynchronous cumulative counters including neutral preroll and post-EOF idle, not an atomic accounting of the 324 source frames. They do not identify where the missing source frames were lost.

All runs use the same local host, text, model, synthetic voice, vendor sample, GPU and LiveKit configuration. They are not independent samples, a stable loss-rate estimate, a browser or physical-presentation measurement, or a concurrency/latency qualification. Instrumentation and unrelated host activity may affect scheduling. No fix or completed delivery gate is claimed. The earlier zero-gap and 12-gap integration observations are retained separately from the latest [instrumented report](../benchmarks/reports/p03-claire-live-preroll.json).

For an already qualified offline environment, the instrumented trial was run from the repository root with a new private output directory:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/runtimes/session-api/.venv/bin/python \
  v2/scripts/qualify_claire_session_worker.py --live-qwen \
  --config v2/var/p04/test-runtime.json \
  --engine v2/var/p03/claire-engine-fp32-n1-final \
  --scene v2/var/p03/claire-render-scene \
  --directory v2/var/p03/claire-receiver-stats-20261028-3 \
  --report v2/var/p03/claire-receiver-stats-20261028-3.json
```

Use a new directory and report name for another run; the qualification preserves previous output. The latest runtime descriptor was pinned with explicit `--allow-receiver-video-gaps`, binding the observed report and existing 183-file runtime inventory without downloading artifacts. The new descriptor was not substituted into either running preview. API/PostgreSQL tests passed **30** cases, including the counter-sanitization check. Next, correlate source capture, RTP/decoder and browser receiver observations under repeated trials without confusing sender acceptance, decoded callbacks and physical display.
