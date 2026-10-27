# V2 validation protocol

Status: implementation specification, reviewed 2026-10-26. Fixtures below are **to be built**, not existing recorded evidence. The only completed measurements are in the [V1 audit](V1_AUDIT.md) and [probe JSON](benchmarks/local-audit-2026-10-25.json). Targets come from the [main plan](../../V2_RESEARCH_AND_BUILD_PLAN.md).

## 1. Reproducible run identity

Every report records source SHA and dirty diff hash, OS/kernel, CPU, GPU SKU/VRAM, driver, CUDA/runtime, model and engine hashes, precision, dependency lock/image digest, power/clock policy, desktop competition, thread limits, avatar manifest, resolution/FPS, audio rates, provider region, network profile and active session count. Record disabled features and fallbacks. Never compare a cropped neural render with a composited/encoded output as if they have the same boundary.

Keep raw event records and a machine-readable summary. Use opaque asset/session identifiers; protect any consented reference media separately. Publish enough configuration and hashes to reproduce results without publishing biometric assets. Record seed and input order. Server-rendered frames must target a future presentation PTS with sufficient compute/transport lead; evaluating the current audible pose before network delivery creates systematic lag. A run with missing dependencies is a failed/unavailable baseline, not a zero-duration measurement.

Illustrative event schema:

```json
{
  "schema_version": 1,
  "run_id": "opaque-run",
  "session_id": "opaque-session",
  "turn_id": "turn-17",
  "generation_epoch": 4,
  "event": "tts.motion_horizon_ready",
  "clock_id": "worker-monotonic",
  "monotonic_ns": 123456789000,
  "audio_start_sample": 0,
  "audio_end_sample_exclusive": 19200,
  "sample_rate_hz": 48000,
  "queue_wait_ns": 2000000,
  "service_ns": 8000000,
  "status": "ok"
}
```

A separate clock-mapping record contains source/target clock IDs, offset, drift estimate, measurement interval and uncertainty. Never subtract timestamps from different clock domains without a mapping. CUDA events measure device service; completed wall time measures queue/launch/transfer effects. Preserve both where meaningful.

## 2. Latency endpoints and cohorts

Annotate the user's final intended speech sample in the **client recording**. Do not move the start to VAD commit. Identify first audible meaningful answer speech and the first corresponding aligned frame at presentation. Report their maximum minus the annotated endpoint, plus audio-only latency and signed A/V skew. A greeting/filler, idle frame or silent packet cannot satisfy useful response.

Collect these intermediate times: capture, transport arrival, ASR partial/final, endpoint commit, LLM request/queue/first token/first useful segment, TTS first PCM and first motion-eligible horizon, motion input readiness/completion, render queue/completion, encode queue/completion, send, receiver decode and presentation. Measure complete-answer duration separately, including stalls and actual spoken duration.

For A2F Claire, explicitly account for the 260 ms future context and the selected initial target/frame interval. Measure time until at least the actual required PCM horizon is available; also report first 100/300/500 ms of PCM. First-byte latency and RTF cannot substitute for this measurement. For direct video, count audio chunk accumulation, all denoising, VAE decoding, crop/composition, encoding and receiver buffering.

Proposed warm reference cohort: preloaded consented avatar/voice, a predeclared English mixture of short/long turns, internal pauses, hesitations, corrections, accents and barge-ins, bounded prompt history, no external tool call, 512²/30 FPS, measured RTT ≤40 ms and stable bandwidth, declared admitted N. The exact input set and network settings must be frozen before the test. Report additional cohorts separately: long context, unusually ambiguous turns, difficult noise, simultaneous speech, tools, other languages, avatar cache misses and poor networks. Do not blend difficult cohorts away or omit failure counts.

Warm readiness means every model/profile/surface is initialized. Cold runs separately measure process launch, model load, engine build if required, avatar load, connection and first playout. Run at least ten independent cold launches and label cache state. Never advertise cached engine startup as installation time.

For final presentation evidence, prefer audio loopback and captured video in one reference clock. A synthetic flash/impulse fixture calibrates the instrumentation, while phonetic footage measures natural content alignment. On browsers, record receiver stats and available presentation callbacks; externally validate the device/browser path. State clock and capture uncertainty. Reject a claimed 10 ms difference if measurement uncertainty is of similar size.

## 3. Required fixture catalog

| Fixture | Input/design | Invariant or assessment |
|---|---|---|
| Sample ledger | Known PCM ramps/impulses at 16/24/44.1/48 kHz, randomized chunk sizes including one-sample and empty chunks | Exact accounting, stateful resampling, bounded delay, no duplicated/missing valid samples |
| EOF | Audio shorter than one window; tail not divisible by hop; empty utterance | Flush once, remove model padding, preserve real final audio; queue-empty is not EOF |
| Epoch cancellation | Cancel before text, during TTS, motion, encode and playback; late callbacks injected | No stale generation accepted; old audible output stops within declared device-buffer bound |
| Timestamp mapping | ±100 ppm clock drift, reconnect, reordered packets, simulated wrap | Monotonic mapped presentation and bounded skew; no cross-epoch interpolation |
| Audio/video calibration | Known audio impulse and visual flash at equal logical PTS | Instrumentation offset and uncertainty established |
| Phonetic quality | Consented bilabials, fricatives, vowels, fast/slow speech, silence and laughter | Closure timing, coarticulation, jaw/teeth/eyes, no idle mouth pumping |
| Turn-taking | Complete sentences; 200/500/1000 ms internal pauses; self-correction; topic change | Human-labeled false commits, endpoint delay, recovery |
| Interruption | Intentional interrupt, brief acknowledgment, cough, keyboard noise and avatar echo | Latency and false-interruption rates, AEC behavior |
| TTS boundaries | Numbers, abbreviations, names, punctuation, sentence fragments and mixed language | No omitted/repeated words, clicks, prosodic resets or normalization errors |
| Identity/view | Held-out temporal clips and supported camera range | Identity, teeth, hair, occlusion and expression stability |
| Session isolation | Distinct voices/avatars, interleaved chunks, duplicate IDs rejected | No cross-session state, output or asset access |
| Overload | Slow TTS, blocked GPU, encoder backlog, network congestion and asset misses | Bounded queues, explicit degradation/rejection, no silent speech deletion |
| Fault/recovery | Worker exit, disconnected client, corrupt asset, failed provider | Explicit error/reset, released resources and new epoch mapping |

Use synthetic fixtures for exact timing invariants, not visual-quality claims. Real quality fixtures need consent, permitted evaluator models and frozen split manifests. Hold out entire recording sessions/identities as appropriate and prevent overlapping context across split boundaries.

## 4. Microbenchmarks and optimization parity

Start with one warm instance and fixed shapes. Preallocate buffers and use inference mode. Record warmup count, sample count and synchronization boundary. For GPU kernels use CUDA events; for a complete operation synchronize its final dependency rather than timing launch alone. Test both isolated and co-loaded operation. Report process/NVML memory as well as framework allocated/reserved peaks; allocator counters omit other runtimes.

Measure TTS generation over short/medium/long utterances, first horizon, sustained audio duration per wall time, cancellation granularity and voice conditioning cache miss. Measure motion across declared lookahead/context, batch sizes and state resets. Measure render separately from conversion, encoding, transfer and presentation, then together. A throughput ratio uses generated media duration divided by elapsed compute time, with the reciprocal RTF clearly named.

Before accepting compilation, quantization or copy removal, compare identical inputs at the event and output levels. Numerical thresholds are model/rig specific: establish a reference error distribution, then choose an absolute tolerance small enough not to change lip contact or eye constraints. A blanket allclose tolerance does not establish perceptual equivalence. Require no new audible boundary artifacts and blinded quality noninferiority.

The checked-in audit probe uses 10 warmups and 100 samples for narrow diagnosis. Its p99 is descriptive only and it does not preserve a complete raw-event trace. Future performance claims require raw samples and the broader protocol; do not extrapolate that probe to production tails.

## 5. Concurrency and statistical treatment

Sweep N = 1, 2, 4, 8, 16, then higher where permitted. For each N run:

1. Shared avatar and unique avatars, with cache-hot and bounded cache-miss variants.
2. Continuous avatar speech; a mixed workload initially 30% speaking/60% listening/10% transition; and synchronized turn-start bursts. Publish realized duty cycles.
3. Staggered session starts plus an open-loop arrival schedule independent of response completion. Record scheduled versus actual starts, offered versus admitted load and rejection rates.
4. At least 30 minutes of steady load, with sufficient turns for the stated percentile. Run eight hours at the proposed admission limit; include repeated attach/detach to detect leaks.
5. Long prefill, simultaneous cancellations, slow provider responses, packet loss and bandwidth changes, both singly and in selected combinations.

Use at least 1,000 turns per p95 cohort and 10,000 for p99 reporting, with confidence intervals; counts alone do not guarantee precision. Bootstrap by session/run blocks to account for correlated samples. Report empirical CDFs, per-session fairness, maximum stalls and sample counts. Repeat qualifying runs under comparable conditions. Do not report only pooled averages.

All scheduled test turns remain in accounting. Timeouts/failures count as SLO violations; report latency quantiles among completed turns separately and use censored bounds where appropriate. Avoid silently discarding failed turns or allowing closed-loop clients to hide overload. Joint endpoint-to-presentation percentiles are measured directly; summing per-stage p95 values is not a statistical estimate of joint p95.

Record first A/V response, complete-answer time, skew, false cutoffs, interruptions, missed frames, underruns, queue ages, CPU/RAM/GPU/VRAM, encoder occupancy, thermal clocks, restarts, rejected sessions and wasted speculation. Admission capacity is the highest repeated N passing all required gates with reserve, not the largest run that avoids OOM.

## 6. Quality evaluation and margins

Model selection uses [independent evidence and Q0](MODEL_QUALITY_BENCHMARKS.md). Distinguish author-run studies, independent human preference, independently executed objective tests and practitioner anecdotes. Paper metrics justify experiments, not promotion. Include NC models equally; no commercial-license penalty. Q0's selection pilot and powered held-out follow-up complement the product-quality gate below; neither a small pilot nor the minimum rater count establishes population superiority.

The user selected Qwen3-TTS for implementation. P02 qualifies its streaming runtime against the official reference; a cross-family TTS comparison is not required to start or complete functional integration. Q0 still qualifies quality claims and final visual/enrollment choices. Pending independent raters stay explicitly pending while unrelated implementation proceeds. Follow the [phase guide](IMPLEMENTATION_PHASES.md) for ordering.

Freeze exact runtime/checkpoint, streaming mode, voice/reference, language and precision. Keep hosted endpoint results separate from local weights, provider-voice Elo separate from controlled-voice Elo, and free-motion video separate from paired reconstruction. Use real user ratings for naturalness; automatic WER, SyncNet and geometry errors remain diagnostic. Retain losing and failed outputs, uncertainty and cohort composition.

Use at least three consenting identities for the first product-quality gate, with varied facial characteristics and speech rates; this is an initial feasibility cohort, not demographic validation. Expand before broad release. Randomize paired clips and blind system labels. Start with at least five independent raters and 20 clips per identity; report rater agreement and confidence intervals and increase sample size if conclusions remain uncertain.

Evaluate identity, mouth/teeth, eyes, temporal stability, gesture appropriateness, speech intelligibility and overall naturalness separately. Use paired pixel metrics only when view/time correspond; for freely generated motion use preferences and anatomical/event metrics. Include mouth/eye crops and full-frame clips. Measure lip offsets over a sweep rather than a single nominal alignment. Licensed embedding/SyncNet-style metrics are corroboration, not sole acceptance.

Proposed optimization noninferiority margins: mean human score reduction less than 0.2 on a five-point scale, no dimension below the main plan's floor, no more than one percentage point absolute intelligibility WER increase, and no more than 5 ms increase in p95 content-alignment error while still passing the absolute skew gate. Apply confidence intervals; insufficient power means inconclusive, not pass. These margins are initial product decisions to calibrate against human perception, not established scientific constants.

Conversational naturalness needs interactive trials in addition to clip ratings. Annotators identify intended turn ends, meaningful interruptions and backchannels before evaluating the system response. Report false cutoff rate together with endpoint latency. A response target achieved by interrupting hesitant speakers fails.

## 7. Acceptance and reporting

M1A requires exact sample/epoch behavior, actual playback alignment, bounded queues and cancellation. M1B additionally needs real retargeted mouth/identity quality, first-horizon timing and memory fit. M2 requires a genuine conversation cohort satisfying p95 <1 s and p99 ≤1.5 s alongside false-cutoff/quality gates, or an explicit measured shortfall and revised profile. M4 requires qualified admission capacity, fault recovery and eight-hour resource stability.

A report contains configuration/rights manifest, fixture hashes, raw events, summary tables with uncertainty, representative protected clips, failures, comparisons, and the resulting ADR/gate decision. Separate observed values, author claims, targets and projections. Do not overwrite the original baseline or silently change a cohort to pass. Retain failed experiments because they explain fallback choices.

## 8. Complete product and deployment qualification

The [implementation backlog](IMPLEMENTATION_BACKLOG.md) defines J01–J10 required journeys; the [release specification](DEPLOYMENT_AND_RELEASE.md) defines operational acceptance. Add these to the media/model checks above:

- Fresh environment and explicit artifact provisioning, actual provider models, real browser playback, approved avatar/voice and no hidden cloud dependency in an all-local claim.
- Complete enrollment from consented capture to evaluation, review, publication and use; duplicate job delivery, crash/retry, cancellation and revocation during publication.
- API tenant isolation, idempotent session creation, stale reservation/fence rejection and no dual audible owner.
- Microphone denial/autoplay block, device changes, reconnect, forced TURN and external networks. Physical speaker/AEC tests supplement headless automation.
- Cancel after the browser has buffered old audio, then resume. Verify no old words reappear after unmute; measure track replacement/resume cost separately when used. Server epoch correctness alone cannot pass this fixture.
- Native publisher access-unit framing, keyframe/rate feedback, actual source timestamp mapping, one-layer congestion behavior and repeated connect/disconnect resource release.
- Active asset revocation with still-valid media tokens, source shutdown, rejoin attempts, cache purge, and restored-backup tombstones. Short token TTL is not proof of active revocation.
- Worker crash, database loss, exhausted GPU/encoder capacity, drain on update, rollback with compatible schema, and isolated backup restore.

Retain separate functional-mock, real-model, photorealistic and hosted evidence. Passing a lower tier cannot satisfy a higher tier. The original synthetic microbenchmarks remain unchanged; this addendum adds no new measured V2 performance.
