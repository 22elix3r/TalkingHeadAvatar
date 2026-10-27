# Codex implementation phases

Updated 2026-10-26. **Start with P00 now.** These are instructions for future implementation; no V2 code or model provisioning was performed while writing this guide. [Main architecture](../../V2_RESEARCH_AND_BUILD_PLAN.md), [work packages](IMPLEMENTATION_BACKLOG.md), [current status](IMPLEMENTATION_STATUS.md).

## Fixed decisions for this implementation

- **Use Qwen3-TTS now.** Start with `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, one English voice, X-Vector conditioning. Keep ICL with an accurate reference transcript as an explicit same-model option. Pin model/tokenizer/runtime revisions during provisioning; do not invent tested hashes here.
- Do not implement PocketTTS, Kokoro, Fish or a TTS provider tournament on the critical path. Those remain future comparison candidates. Qwen is user-selected, not declared an independently proven universal winner.
- First qualify the official Python implementation as an offline reference, then a real incremental PCM runtime for the same model. The output contract is stable even if its backend changes. See the runtime decision below.
- One local session first. CPU ASR/VAD/turn detection; Qwen TTS on GPU; one media owner when introduced. Start LLM on CPU with bounded context, or an explicitly configured remote provider. Test GPU LLM co-residency later. Do not assume a 16 GB card fits every component.
- Use A2F Claire plus a research-eligible rig/Gaussian implementation as the first visual integration baseline. StreamingTalker and direct-video alternatives are conditional follow-ups to a failed visual gate, not mandatory integrations before the first demo.
- Keep React/TypeScript/Vite, FastAPI, PostgreSQL and LiveKit as selected application technologies. Add each when its phase needs it. No Kubernetes, new SFU, foundation-model training or custom browser Gaussian renderer in the first slice.
- Preserve V1, checkpoints and unrelated working-tree changes. All new runtime work belongs under `v2/`. Noncommercial research components remain eligible; commercial clearance is deferred.

## How to run each phase with Codex

Use the phase prompt below. Codex should inspect the current tree and status, implement the named phase, run its relevant checks, fix failures, and update the status with evidence. It should make routine implementation choices without repeatedly asking for approval. A phase boundary is a reviewable result, not permission to claim the whole product is finished.

If the phase exceeds one session, stop only with a precise checkpoint: completed tasks, changed paths, commands/results, actual blocker and next executable action. A later “continue” resumes that phase. Do not redo passed work or reopen the model survey without a concrete failure. Do not commit, deploy externally or change global settings merely because a phase mentions them.

Inputs needing the user's own consented voice/avatar or unavailable hardware are explicit dependencies. Continue independent implementation with synthetic fixtures while documenting the missing input; never mark real-model, real-asset or subjective-quality acceptance passed by substitution. Synthetic fixture status must be visible in reports.

## Phase map

| Phase | Observable result | Work packages / older milestone | Dependency |
|---|---|---|---|
| P00 | Isolated, runnable V2 foundation and read-only diagnostics | E00, minimal E01 / M0 | None; start now |
| P01 | Correct sample clock, cancellation and replay contracts | E02–E03 / M0–M1A | P00 |
| P02 | Qwen text → real incremental audio → local playback | E05, audio part of E04 / M1A–M1B | P01; explicit model provisioning |
| P03 | One real avatar speaks Qwen audio on one timeline | E06–E07, visual E04 / M1B–G1 | P02; research-eligible rig/avatar |
| P04 | Browser microphone → interruptible audiovisual answer | E09, minimal E10–E11 / M2 | P03 functional media gate |
| P05 | Enroll, approve, use and revoke an avatar/voice | E10–E13; early E17 / M3 | P04; quality-qualified enrollment recipe |
| P06 | Native GPU media, reproducible local deploy and virtual devices | E08, E14, E19 / M4 | P04; integrate P05 before release |
| P07 | Measured concurrency, admission and operational evidence | E15–E16 / M4 | P05–P06 |
| P08 | Qualified deployable research release with recovery/security | E17–E18 / M6 | P05–P07 and quality acceptance |
| P09 | Optional client-rendered tier | E20 / M5 | Stable exported assets and P08; not required for server release |

P00–P04 are the shortest route to a real conversation. P05–P08 complete the requested deployable system. Existing E/J IDs stay authoritative for coverage. The P phases define execution order; the older M milestones describe outcomes, not a second competing task sequence.

## P00 — Foundation, without models or services

**Read:** root AGENTS, main plan sections 1/6/17, V1 audit summary, and this guide. Inspect actual current files before editing.

**Implement:** `v2/pyproject.toml`, isolated Python 3.12 environment/lock when compatible versions resolve, `src/talkinghead_v2/`, test layout, a tiny typed config and CLI, `v2/README.md`, `v2/docs/baseline.md`. Add read-only `tha doctor --profile local` diagnostics for Python/environment, CUDA/device/encoder availability, cached artifact presence and writable output space. Missing optional GPU tools produce a clear diagnostic, not an import crash. Record the V1 load/test failures already known; no need to repeat expensive probes simply to scaffold V2.

**Keep small:** no torch requirement for CPU contract tests; separate optional inference dependencies. No browser shell, database, container stack, native CMake boilerplate or empty provider classes yet. E01's browser/native deliverables arrive when their phases require them.

**Acceptance:** editable installation in the isolated environment, CLI help and doctor work; CPU smoke/config validation checks pass offline; importing the core does not initialize CUDA or fetch models. Document exact working setup/check commands. The status identifies P01 as next.

**Prompt:** `Implement P00 from docs/v2/IMPLEMENTATION_PHASES.md. Follow AGENTS.md, preserve V1, validate the result, and update IMPLEMENTATION_STATUS.md. Complete P00 only; do not download model weights.`

## P01 — Clock and cancellation before neural inference

**Implement:** typed `TextSegment`, `AudioChunk`, `MotionFrame`, capabilities, EOF/discontinuity and epoch events; integer sample ledger; stateful resampling; queues bounded by media duration; generation cancellation and deterministic PCM replay. Use in-process contracts first; no remote RPC until needed. Provide an executable replay fixture with irregular chunks, two rates, silence, a partial final chunk and cancel/restart.

**Acceptance:** sample conservation and resampler tails; no unexplained gaps/overlaps; stale-epoch callbacks cannot reach the sink; bounded backpressure does not delete speech; EOF and cancellation behave differently. Tests must exercise faults and boundary cases rather than mirror object constructors. Test synthetic playback scheduling separately from actual device presentation. No claim of sub-second conversation.

**Prompt:** `Implement P01 from docs/v2/IMPLEMENTATION_PHASES.md using the existing P00 foundation. Complete and validate its clock/replay/cancellation contracts, then update IMPLEMENTATION_STATUS.md.`

## P02 — Qwen3-TTS working end to end

**Implement in this order:**

1. Explicit artifact provisioning with size/hash verification and immutable cache; registry entries for the Base model, tokenizer and approved reference voice. Startup consumes the cache and does not download. Use existing cache where valid. A consented reference is required for human voice cloning; synthetic reference audio can exercise plumbing with that limitation recorded.
2. Official Python offline adapter for quality reference. Persist native-rate PCM and metadata; do not call it streaming. Begin with the model's unquantized supported CUDA dtype. Capture environment, warm/cold time and peak allocated/reserved/device memory before adding other GPU workers.
3. Streaming adapter with actual early PCM callbacks. **First bounded backend spike:** inspect and build the current `maxhorizons/qwen3-tts-cpp-streaming` (formerly `offgridai`) CUDA/F16 Base route in isolation on Fedora, pin source/GGML/converter revisions, and verify 1.7B conversion plus speaker conditioning against the Python reference. If it cannot support the required model/conditioning, callback or cancellation boundary, document that failure and try the X-Square Base route below. Do not implement both service architectures at once.
4. Connect the successful backend to P01 and local audio output. Accept committed text segments even if the backend requires complete text per segment; declare `incremental_text_input=false` in that case. Incremental **audio output within each segment** is mandatory for this phase's streaming acceptance. Joining a whole generated waveform and slicing it afterward fails.
5. Preload one voice and model. Cancellation rejects late callbacks immediately; the worker acknowledges when compute has stopped. Use bounded cooperative cancellation if supported; worker restart is a documented recovery fallback with a cold-readiness cost, not invisible normal operation.

**Contract:** expose `open_session`, ordered text input, async audio events, flush, cancel and close behind the existing provider interface. Adapt actual runtime APIs after inspection; these project names are not claimed upstream APIs. Include voice/model/runtime identity, native sample rate and contiguous sample ranges. Keep text normalization and speaker cache keys explicit; preserve spoken punctuation/numbers and privacy boundaries.

**Acceptance:** actual audible streaming before synthesis completes on a long utterance, first 100/300/500 ms horizons measured, no duplicated/truncated speech at segment boundaries, final codec tail accounted for, cancellation followed by clean new speech, and 30-minute sequential-request stability. Test numbers, abbreviations, Unicode, very short text, pauses and long text; record WER/human listening and runaway/repetition failures. Compare the converted runtime with official-reference clips; numeric waveform equality is not a required quality claim.

Record TTFA, maximum PCM production gap, sustained RTF and peak memory on the 4070. Targets remain RTF <0.7 and motion-horizon p95 ≤240 ms for the eventual low-latency profile; failure records a performance shortfall and prioritizes a same-model runtime fix. It does not authorize relabeling batch output, switching to Pocket, or claiming the target passed. Preserve Qwen if a separate GPU/explicit remote Qwen deployment is needed. A 0.6B Base or quantized Qwen variant is an explicit measured fallback with its own quality gate, never an unnoticed substitution.

**Prompt:** `Implement P02 from docs/v2/IMPLEMENTATION_PHASES.md with Qwen3-TTS as the selected provider. Inspect cached artifacts and the documented backends, build one real streaming path, measure its limitations honestly, and update IMPLEMENTATION_STATUS.md. Do not add other TTS model families.`

### Runtime evidence behind P02

Accessed 2026-10-26: [official Qwen source](https://github.com/QwenLM/Qwen3-TTS), [Base weights/card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base). These establish model identity and the offline reference; verify the pinned wrapper's yield behavior rather than assuming its architecture-level streaming claim describes its Python API.

[Native C++ runtime](https://github.com/maxhorizons/qwen3-tts-cpp-streaming) documents Base 1.7B incremental PCM and cloning, F16/GGML conversion and optional CUDA. Its README emphasizes Windows commands and RTX 5090 measurements; Fedora support, memory, cancellation and quality parity remain our qualification tasks. This is an implementation choice to test, not a verified benchmark on our hardware.

[X-Square runtime](https://github.com/X-Square-Robot/Qwen3TTS-Streaming) is the second backend candidate. Its v0.2 README marks Base/X-Vector and ICL experimental; its stable validation concerns CustomVoice. It requires export/build artifacts and documents a later concurrency regression relative to historical numbers. Do not silently change to CustomVoice to inherit those claims. Inspect scripts before use: its setup can install an environment and download artifacts. Use a project-isolated environment/container, not unattended global setup. Neither candidate's runtime-author measurements count as independent confirmation of our quality or latency.

## P03 — First speaking avatar

**Implement:** one actual compatible avatar/rig; A2F adapter and explicit retarget calibration; a timestamped research Gaussian renderer and mesh timing fallback; bounded idle/listening policy; local aligned audio/video output. Reuse eligible V1/third-party components behind isolated adapters; never repair a topology mismatch by truncating vertices or padding coefficients. Use recorded Qwen PCM first, then the live P02 stream. No expensive training yet.

**Acceptance:** measured required lookahead including Claire's 260 ms future audio; bilabial closure/teeth checks; original identity; independent render cadence; p95 presentation skew ≤40 ms and stale-audio suppression ≤200 ms over the defined fixture, with 30-minute bounded queues. Measure Qwen+motion+render peak memory together. If the rig is unavailable, timing tests may proceed but the real-avatar phase remains incomplete.

**Quality gate:** separate technical operation from subjective acceptance. Assemble Q0 clips for independent raters; use StreamingTalker or one direct-video challenger only if the baseline fails a documented quality criterion. Rater scheduling does not block P04's functional integration, but do not mark visual quality passed or commit to expensive enrollment training until Q0/G1 resolves it. Q0 does not reopen the selected TTS family.

**Prompt:** `Implement P03 from docs/v2/IMPLEMENTATION_PHASES.md using Qwen audio and one real avatar. Prove clocks, cancellation, retarget quality and co-loaded memory; document unavailable assets and update IMPLEMENTATION_STATUS.md.`

## P04 — Browser conversation

**Implement:** thin React device/playback UI, FastAPI session endpoints, minimal PostgreSQL session/asset metadata, local-only development identity, LiveKit raw-frame reference publisher and receiver-safe cancel/resume. Wire CPU streaming ASR, VAD/turn policy, bounded-context LLM and stable text segmentation into the session actor and Qwen. Configure any remote provider explicitly. Capture mic permission, autoplay, captions, stop/mute, reconnect and error recovery.

**Acceptance:** a real user speaks and hears/sees a meaningful answer; pause/hesitation and interruption fixtures pass without stale words reappearing. Measure end-of-user-speech through aligned browser presentation, with endpoint patience included. Record p50/p95/p99 and false cutoffs; an SLO miss is not hidden by changing boundaries. Developer identity stays loopback-only and must not enable public hosting. Existing P03 visual evidence stays attached to the run.

**Prompt:** `Implement P04 from docs/v2/IMPLEMENTATION_PHASES.md. Deliver the real browser conversation using the existing Qwen/media path, test interruption and recovery, and record measured latency and pending quality gates.`

## P05 — Enrollment and asset lifecycle

**Implement:** finish API/database semantics, consent/reference records, validated uploads, immutable assets, job stages, held-out QC, review/publish UI, cache authorization and revocation. Start with import of an authorized compatible asset; implement one selected enrollment recipe after G1, not every research reconstruction model. Keep enrollment GPU work separate from active serving. Add tenant scoping and access-denial tests now, before any hosted use.

**Acceptance:** real capture/import → approve → select → converse → revoke; retries cannot double-publish; revocation wins concurrent enrollment/session attachment. At least three consented identities pass the declared quality gate before a multi-avatar quality claim. Qwen voice records and cached conditioning follow the same lifecycle. Missing consented captures remain explicit pending evidence.

**Prompt:** `Implement P05 from docs/v2/IMPLEMENTATION_PHASES.md and journeys J02/J07. Complete one enrollment path and asset/voice revocation with real evidence; preserve research eligibility and update status.`

## P06 — GPU media and runnable local deployment

**Implement:** LiveKit C++ encoded publisher with CUDA conversion/NVENC and explicit surface ownership; GStreamer fallback only if the primary integration fails. Retain the measured raw-copy path as reference. Build pinned images/Compose and `doctor/provision/up/down`; provide Fedora setup/runbook and virtual camera plus microphone adapter. Review dependencies and driver/runtime compatibility before rebuilding engines. A phase instruction is not authorization for a global driver replacement.

**Acceptance:** no raw-frame readback claim without a trace; sample/PTS mapping, keyframes, connect/disconnect and GPU resource lifetime tests; fresh local provision/start; actual conference-app virtual-device loop; declared copies and fallbacks documented. No public deployment before P08 security qualification.

**Prompt:** `Implement P06 from docs/v2/IMPLEMENTATION_PHASES.md. Qualify native media, packaging and virtual devices against the reference path, and record exact runnable commands and resource evidence.`

## P07 — Capacity and operations

**Implement:** shared-weight pools, per-session Qwen state, fenced reservations, deadline-aware bounded queues, admission, traces/metrics, failure recovery and runbooks. Begin with one admitted session and increase only after measurement; no dynamic batching until the non-batched path is correct. Qualify GPU LLM co-residency here if desired.

**Acceptance:** 1/2/4/8/16-session sweep as hardware permits, shared/unique assets, bursts, cancel, cold misses, worker death, fairness and eight-hour soak at admitted capacity. Record rejected/failed requests and the stopping constraint; quality and tail latency must pass together. No fabricated 16-session pass or cost figure from peak FPS.

**Prompt:** `Implement P07 from docs/v2/IMPLEMENTATION_PHASES.md. Add measured admission and observability, run feasible load/fault cohorts, and publish actual capacity and limitations.`

## P08 — Deployable research release

**Implement:** OIDC/roles, scoped service/media tokens, active revocation and rejoin defenses, TLS/TURN, secret handling, migrations, drain/rollback, signed manifests, backup/restore, retention/deletion and operator documentation. Review security controls already introduced in P05; do not postpone basic tenancy until this phase. External deployment requires the actual target and authorization, not an invented cloud account.

**Acceptance:** all required J01–J10 journeys linked to evidence, real Qwen and actual browser/avatars, external-network forced TURN, tested restore with revocation tombstones, independent quality acceptance and declared latency/capacity profile. Commercial rights may remain deferred for this research release. Missing runtime, subjective evaluation, recovery or security evidence must remain visible; a mocked or localhost-only stack is not a release pass.

**Prompt:** `Implement P08 from docs/v2/IMPLEMENTATION_PHASES.md. Close the research release gates with actual evidence, retain declared performance limitations, and update IMPLEMENTATION_STATUS.md and operator runbooks.`

## P09 — Optional client rendering

Implement timestamped motion delivery, mesh then Gaussian parity, own-audio-playhead synchronization and device/thermal/privacy qualification. This phase is optional and cannot delay a qualified server-rendered research release. Use E20/J11; add no universal mobile support claim.

**Prompt:** `Implement optional P09 from docs/v2/IMPLEMENTATION_PHASES.md for the selected client devices. Validate J11 and keep this tier's quality, latency and capacity evidence separate.`
