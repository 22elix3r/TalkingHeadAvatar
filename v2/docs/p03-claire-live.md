# Live Qwen and Claire co-load observation

Local functional observation, 2026-10-27. [Evidence](../benchmarks/reports/p03-claire-live.json) binds the actual cached Qwen runtime/model, Claire engine/SDK, bridge, native worker, source modules and scene. This is one known-text utterance through a Python LiveKit receiver, with a synthetic voice reference. It is not browser conversation, physical lip-sync, independent quality, capacity or soak acceptance.

Actual command from the repository root:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/runtimes/session-api/.venv/bin/python \
  v2/scripts/qualify_claire_session_worker.py --live-qwen \
  --config v2/var/p04/test-runtime.json \
  --engine v2/var/p03/claire-engine-fp32-n1-final \
  --scene v2/var/p03/claire-render-scene \
  --directory v2/var/p03/claire-session-live-first \
  --report v2/benchmarks/reports/p03-claire-live.json
```

The preloaded Qwen provider produced **258,645 native samples at 24 kHz**, in **15 actual callbacks**. The paired FIR emitted **517,290 playback samples / 172,430 motion samples**. **485 paired packets were sent before the TTS request completed**; this exercises actual incremental model output, not a previously completed waveform split into chunks. From the controller's request start, first received PCM was **259.16 ms**, first paired packet **292.48 ms**. Request wall time includes downstream backpressure and is not isolated TTS throughput or end-user latency. Native and controller timing fields remain distinct in the report.

Claire/NVENC/LiveKit published **324 source frames / zero source-video skips**, exact valid audio plus **1,110 codec padding samples**, and explicit source EOF with `presentation_ack=false`. Host audio peak was **48,000 samples**, host motion **26 packets**, GPU poses **16 frames**. One keyframe demand and ten rate updates were observed. The receiver decoded **316 source frames** plus eleven post-EOF idle frames and nonsilent audio. Missing source indices were **0, 1, 2, 221, 223, 226, 228, 233**. The final source frame arrived; startup and five later receiver gaps remain findings. A checked receive-queue overflow would fail the run rather than silently discard incoming speech.

Sequential NVML snapshots targeted 100 ms, with **116 samples / zero query failures**, including **89** snapshots with both Qwen and Claire allocations present. Sampled maxima: total device **6,957 / 16,376 MiB**, Qwen process **4,392 MiB**, combined Claire/render/NVENC/LiveKit worker **1,203 MiB**, receiver process **746 MiB**. The largest same-query Qwen+Claire sum was **5,595 MiB**. These are sampled observations, not guaranteed loading/allocator peaks, a model-pool capacity or an eight-hour stability result. The combined media worker is broader than the plan's isolated encoder budget, and does not pass that 512 MiB gate by comparison. Desktop/driver allocations remain included in total-device use.

The helper shares the admitted device lease across both children and closes them before releasing admission, including startup failures. Private diagnostics/memory samples remain under the cohort directory; no raw generated voice or likeness media is published. No model/runtime download, training, global dependency/driver change or external deployment occurred. The normal preview remains ready on its existing backend.

Next: provide explicitly tagged neutral preroll before speech to qualify receiver startup, investigate remaining receiver gaps, and integrate an opt-in vendor research actor/profile with fresh-track cancellation. Physical A/V, latency/false-cutoff cohorts, Gaussian retarget, human enrollment/consent, independent quality and release remain open. P03/P06 remain in progress.
