# Receiver startup with neutral preroll

Local observation, 2026-10-27. The [first live co-load](p03-claire-live.md) missed source frames 0–2. The native worker now publishes explicitly tagged source-matched neutral geometry before speech, at the same CPU audio clock's 30 Hz grid. It requires at least three accepted preroll captures and 100 ms of clock time before beginning speech. Speech frame 0 requests its own IDR. The neutral upload is **805,464 bytes once at owner initialization**, not a dynamic neural geometry readback. Dynamic speech poses remain owned GPU copies.

Preroll has **no source audio/neural timestamp**. The bridge returns an unavailable-source sentinel and the worker omits those fields from preroll metadata, setting `idle_before_audio=true`. Preroll never consumes a queued speech pose or increments neural/source-frame counters. Once speech begins, neutral preroll is forbidden. Source/presentation epochs, PTS and EOF idle remain separately identified. Audio startup waits for successful capture acceptance, not an attempted publication.

Actual cached-GPU commands use the same engine, scene, input and reference flags as the [bridge](p03-claire-bridge.md) and [session](p03-claire-session.md) runbooks:

```sh
cmake --build v2/var/build/native-media --parallel 4 \
  --target tha_claire_session_media tha_claire_bridge_probe
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/.venv/bin/python \
  v2/scripts/qualify_claire_bridge.py \
  --engine v2/var/p03/claire-engine-fp32-n1-final \
  --scene v2/var/p03/claire-render-scene \
  --input-directory v2/var/p03/claire-qwen-input \
  --reference-directory v2/var/p03/claire-direct-render-final \
  --directory v2/var/p03/claire-bridge-preroll-final \
  --report v2/benchmarks/reports/p03-claire-bridge.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/runtimes/session-api/.venv/bin/python \
  v2/scripts/qualify_claire_session_worker.py \
  --config v2/var/p04/test-runtime.json \
  --engine v2/var/p03/claire-engine-fp32-n1-final \
  --scene v2/var/p03/claire-render-scene \
  --directory v2/var/p03/claire-session-preroll-qualified \
  --report v2/benchmarks/reports/p03-claire-session-worker.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/runtimes/session-api/.venv/bin/python \
  v2/scripts/qualify_claire_session_worker.py --live-qwen \
  --config v2/var/p04/test-runtime.json \
  --engine v2/var/p03/claire-engine-fp32-n1-final \
  --scene v2/var/p03/claire-render-scene \
  --directory v2/var/p03/claire-session-preroll-live-qualified \
  --report v2/benchmarks/reports/p03-claire-live-preroll.json
```

Warnings-as-errors build, **eleven bridge cases**, **thirteen sender/fault cases**, and **one live cached-Qwen case** passed. The bridge's added case emits three neutral frames followed by all 84 speech frames, proves that neutral consumes no source state, and rejects preroll after speech. Non-preroll normal/paced/irregular/fresh-owner decoded speech still equals the previous direct reference exactly. CBR encoder history differs in the separate preroll profile; exact preroll-profile pixel parity is not asserted.

The recorded-Qwen receiver decoded **84/84 source frames with no missing indices**, plus three received neutral frames. All short-EOF/fault cases passed with the startup gate. Blocked-producer worker retirement was **376.22 ms**, not a receiver suppression result.

The final live cohort decoded **324/324 source frames, including 0 and the last pose**, plus ten neutral and eleven post-EOF idle frames. Sender conserved **517,290 valid samples + 1,110 padding**, produced 324 source frames and zero source-video skips. It published fourteen neutral frames before speech. Host queue peaks stayed 48,000 audio samples / 26 motion packets and 16 GPU poses. One keyframe request and twelve rate updates were observed. The preceding preroll runs also received source frame 0 but had eight and three later missing frames, respectively; these remain private and are not erased by the final zero-gap run. This is local functional evidence, not a stable loss-rate or percentile qualification.

Final live NVML: **112 samples, zero query failures, 91 co-residency samples**. Sampled device maximum **6,957 / 16,376 MiB**, Qwen **4,392 MiB**, Claire media **1,203 MiB**, receiver **746 MiB**; same-query Qwen+Claire maximum **5,595 MiB**. These are sampled maxima and do not establish capacity/soak or the isolated encoder budget. [Live preroll evidence](../benchmarks/reports/p03-claire-live-preroll.json), [protocol evidence](../benchmarks/reports/p03-claire-session-worker.json) and [bridge evidence](../benchmarks/reports/p03-claire-bridge.json) bind the actual source/runtime/input identities.

No new model/runtime download, training, global setting/driver change, V1 mutation or cloud deployment occurred. The existing normal preview remains unchanged. Next: integrate an opt-in vendor research actor/profile and exercise typed/microphone browser journeys, fresh-track cancellation, startup and actual A/V. Human enrollment, Gaussian retarget, independent quality, latency and release gates remain open; P03/P06 remain in progress.
