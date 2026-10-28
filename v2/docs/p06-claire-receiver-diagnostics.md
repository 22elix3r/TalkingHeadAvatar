# Claire receiver frame diagnostics

Local observations, 2026-10-28. The live-Qwen/Claire qualification now records received source-frame IDs and callback-arrival offsets, the checked receiver queue's high-water marks, and a sanitized snapshot of the SDK's inbound video RTP counters. It exports no track, transport, candidate or session identifiers from that snapshot. Stats failures are reported as unavailable; they do not become zero-loss observations. [Aggregated evidence](../benchmarks/reports/p06-claire-receiver-diagnostics.json) retains seven same-utterance runs and source/binary identities.

The earlier integration run observed **312/324** source frames. Three uninstrumented repeats each observed **324/324**. Three subsequent instrumented runs observed **322/324, 322/324 and 323/324**, with no source-video skips reported by the sender. Every run received the first and final source frames. In the instrumented runs, the checked video queue peaked at one, two and one frames out of its 30-frame bound. The SDK snapshots each reported zero packets lost or discarded, zero NACK/PLI/FIR and zero `frames_dropped`; they reported 352 frames received and 350 decoded. These are asynchronous cumulative counters including neutral preroll and post-EOF idle, not an atomic accounting of the 324 source frames. They do not identify where the missing source frames were lost.

All runs use the same local host, text, model, synthetic voice, vendor sample, GPU and LiveKit configuration. They are not independent samples, a stable loss-rate estimate, a browser or physical-presentation measurement, or a concurrency/latency qualification. Instrumentation and unrelated host activity may affect scheduling. No fix or completed delivery gate is claimed. The earlier zero-gap and 12-gap integration observations are retained separately from the latest [instrumented report](../benchmarks/reports/p03-claire-live-preroll-20261028-rtp.json).

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

## Source capture correlation

The qualifier can now opt in to `--trace-source-captures` for a live-Qwen trial. The worker keeps at most 3,600 speech-frame records in memory and flushes them to its private stderr only when the owner retires. Each record contains a frame ID, source sample, nominal PTS, source-capture call times, encoded byte count and keyframe flag; it contains no audio, image or identity. The normal API removes the trace environment variable from its child process. The parser bounds log size, validates ordering/completeness and checks each received source frame against its accepted source ID.

Three traced trials observed **323/324, 324/324 and 321/324** source frames. The missing source indices were 228 in the first trial and 215, 222 and 224 in the third. All 324 source captures were accepted by the encoded-source API in each run; the missing IDs have corresponding successful captures. The largest observed capture-call durations were **86, 79 and 103 microseconds** respectively. These times and nominal PTS use the native process's steady clock; they do not measure packet transmission or screen presentation. SDK source acceptance is an earlier boundary than delivery, so the missing frames' downstream location and cause remain unresolved. The first trial used the preceding parser; the later two additionally checked received IDs against captures.

The final traced run used:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/runtimes/session-api/.venv/bin/python \
  v2/scripts/qualify_claire_session_worker.py --live-qwen --trace-source-captures \
  --config v2/var/p04/test-runtime.json \
  --engine v2/var/p03/claire-engine-fp32-n1-final \
  --scene v2/var/p03/claire-render-scene \
  --directory v2/var/p03/claire-capture-trace-20261028-3 \
  --report v2/var/p03/claire-capture-trace-20261028-3.json
```

The unchanged 19-case native protocol/fault cohort passed with tracing disabled, receiving **84/84** recorded-Qwen source frames. Offline core tests passed **335**, and API/PostgreSQL tests passed **31**. A new private descriptor binds the traced runtime and the last imperfect live observation with explicit gap opt-in; neither running preview was changed. See the [capture summary](../benchmarks/reports/p06-claire-source-capture.json), [detailed trace](../benchmarks/reports/p03-claire-live-preroll-20261028-source-trace.json) and [untraced worker report](../benchmarks/reports/p03-claire-session-worker-20261028-capture.json). The trace, receiver timestamps and RTP snapshots do not establish a browser/physical A/V, stable delivery or latency pass.

## Native queue and sender counters

The checked Python video queue is not the first receiver queue. The installed Python SDK constructs a native stream without setting its native queue limit. Separately inspected, pinned LiveKit Rust source defaults that queue to one frame and can evict the oldest frame; the Python wheel's exact source identity with that checkout is not established. The qualification now explicitly requests a native queue of either one or eight frames, bounds the checked Python queue separately, and records only the SDK's periodic native-overflow warning count. The SDK does not echo the configured size or expose a native queue high-water/drop counter through this interface, so a missing warning is not proof of no overflow.

Three alternating pairs received **319, 323 and 322 of 324** source frames with a one-frame request and **324, 322 and 320 of 324** with an eight-frame request. No periodic native-overflow warning was observed before the receiver snapshot. These six local runs do not establish a loss rate or prove which queue caused a gap; the larger request did not consistently eliminate it. The sender's asynchronous RTP snapshots were unchanged after a 300 ms qualification-only pause, but their totals include accepted preroll and idle frames and cannot be attributed to individual source frames. The [queue-study report](../benchmarks/reports/p06-claire-native-queue-study.json) records the exact counters and source identities.

The final queue-comparison trial used:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/runtimes/session-api/.venv/bin/python \
  v2/scripts/qualify_claire_session_worker.py --live-qwen --trace-source-captures \
  --native-video-queue-frames 8 --config v2/var/p04/test-runtime.json \
  --engine v2/var/p03/claire-engine-fp32-n1-final \
  --scene v2/var/p03/claire-render-scene \
  --directory v2/var/p03/claire-nativequeue-8-rep-3-20261028 \
  --report v2/var/p03/claire-nativequeue-8-rep-3-20261028.json
```

That cohort also exposed unbounded native audio input. The later
[bounded-input qualification](p06-inbound-audio.md) and
[preview upgrade](p06-inbound-upgrade.md) address that separate boundary;
they do not change the historical receiver observations here.

The current untraced **19-case** native worker/fault cohort passed its existing functional checks, with **83/84** recorded-Qwen source frames observed and one missing intermediate frame. It remains evidence of a receiver gap, not a full-delivery pass. Isolated API/PostgreSQL tests passed **32**; the descriptor was repinned to the latest live observation with explicit receiver-gap opt-in. See the current [worker](../benchmarks/reports/p03-claire-session-worker.json) and [live report](../benchmarks/reports/p03-claire-live-preroll.json).

## Source-matched FFI boundary and decoder comparison

The next qualification explicitly selected the already-built FFI
`16eb6e043651eb1c509bf515f512abac2c0e079035cc24928f72d3daa9866d03`
through descriptor
`ec5d27245548829391ab745b136b4a7899d020cc251a2c771802edbbe6afe233`.
Its pinned Rust source is `bae4df2f3d6c8f0f6ce81a7370c445779f1948d2`,
with the separately qualified audio fail-closed patch. This removes the earlier
wheel/source-identity uncertainty for these new receiver trials. The receiver's
native audio queue is explicitly 25 decoded frames; its separate checked
Python queue is 30 output frames at 10 ms. This benchmark receiver differs
from the serving microphone's 12 × 20 ms Python queue.

An opt-in, bounded metadata observer records source frame IDs at the FFI
callback before Python dispatch. It always forwards the original event,
validates the expected generation, rejects malformed/oversized observations
and retains at most 3,600 source records per stream. It exports no raw metadata,
images or session/track identifiers. Source captures, callback IDs and consumer
IDs are checked against each other. Explicit SDK disposal happens while the
event loop remains alive, and the batch runner checks the entire process exit.

The pinned source supports `LK_DISABLE_NVDEC`. An explicit qualification-only
option selects that receiver factory behavior; the external NVENC publisher's
child environment removes the flag. Sanitized RTP stats identify the observed
decoder as `nvidia_h264` or `ffmpeg`, without exporting arbitrary implementation
strings. The previews and browser decoder selection were not changed.

The second alternating matched cohort observed:

| Receiver decoder | Requested native video queue | Source frames at FFI and consumer, two repeats |
|---|---:|---|
| NVIDIA H.264 | 1 | 324/324; 324/324 |
| FFmpeg | 1 | 324/324; 324/324 |
| NVIDIA H.264 | 8 | 324/324; 322/324 |
| FFmpeg | 8 | 324/324; 321/324 |

All nine processes, including a deliberate 500 ms callback-stall control,
exited 0 without panic. Every source frame observed at the FFI callback reached
the Python consumer. The two normal gaps were already absent at that earlier
boundary. Neither decoder family nor a larger queue consistently eliminated
gaps: the preceding nine-case cohort also had 319/324 with NVIDIA/queue 1 and
318/324, 310/324 with FFmpeg/queue 8. Those are same-host observations, not
independent samples or a stable loss-rate estimate.

The second cohort's deliberate stall measured **500,059,479 ns**, produced a
native overflow warning and received **298/324** source frames. Its first
fourteen missing indices follow the injected stall, while twelve later gaps
remain separately visible. This positive control shows that the observer can
detect induced native queue overflow. A periodic warning counter is still not
an exact native drop counter or universal proof that no overflow occurred.

Sender aggregate accepted/encoded/sent totals also differ. For example, one
normal gap case reported 357 accepted versus 353 encoded/sent, and the control
reported 358 versus 343. These totals include neutral preroll and post-EOF idle;
they cannot assign an individual missing source frame to an encoder, network
or decoder stage. The sender counters were unchanged after the qualification's
300 ms settling pause. This makes the accepted-source → encoder-output boundary
the next useful trace target; it does not establish a loss cause.

Review tightened batch ownership: timeout, nonzero exit, signal death and
premature success with surviving descendants now fence the owned process group
before reaping the leader, then verify actual descendant termination. This
prevents a failed qualifier from stranding its Qwen/Claire workers or signaling
a reused group ID. Five real subprocess tests cover these paths, including a
descendant that ignores SIGTERM and holds a file lock. An initial timeout test
assumed immediate lock release after SIGKILL; it now waits for actual release.
The earlier successful cohorts did not invoke the subsequently corrected
failure paths and remain separately identified in the report.

With the final runner, a fresh **five-process** recheck (`--repeats 1`) completed
with exit 0/no panic: NVIDIA/queue 1 **318/324**, FFmpeg/queue 1 **323/324**,
NVIDIA/queue 8 **323/324**, FFmpeg/queue 8 **324/324**, and the induced-overflow
control **309/324** with a warning. Again, no observed FFI source callback was
lost at the consumer. The exact command, from `v2/`, was:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 runtimes/session-api/.venv/bin/python \
  scripts/qualify_claire_receiver_matrix.py --repeats 1 \
  --receiver-runtime var/p06-claire/inbound-runtime-reason-v1.json \
  --config var/p04/test-runtime.json --engine var/p03/claire-engine-fp32-n1-final \
  --scene var/p03/claire-render-scene \
  --directory var/p06-claire/receiver-source-matrix-fenced-final-20261028 \
  --report var/p06-claire/receiver-source-matrix-fenced-final-20261028.json
```

Use new paths for another cohort. The default `--repeats 2` runs nine processes;
the final one-repeat run rechecks the current ownership guard with all four
decoder/queue settings and the positive control.
[Sanitized evidence](../benchmarks/reports/p06-claire-source-matched-receiver.json)
retains all three cohorts, each runner revision, child receipt hashes, exact
runtime/model/scene identities, missing indices and aggregate counters.

The untraced **19-case** protocol/fault regression also passed with the explicit
source-matched receiver, observing **84/84** recorded-Qwen source frames in
[its report](../benchmarks/reports/p03-claire-session-worker-source-matched-20261028.json).
Final offline suites passed **356 core** and **45 API** tests. No model/software
download, native sender rebuild, preview mutation or V1 change occurred.
Physical A/V, complete live delivery, independent quality, capacity and release
remain open; no phase or decoder-winner claim is made from these diagnostics.

The subsequent [publisher-stage comparison](p06-publisher-stage.md) matched
the observed missing IDs to the WebRTC upload/output handoff and qualified a
default-off external-rate-control candidate in short local fixtures. It is not
enabled in the serving previews; lifecycle, rate-pressure and browser rollout
qualification remain separate next steps.
