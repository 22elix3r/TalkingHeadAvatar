# TalkingHeadAvatar development guidance

## Project authority and scope

`main` contains V2. The V1 runtime and original history are preserved on `v1/legacy`; run the V1-only checks below from a separate V1 worktree. Preserve ignored legacy caches and unrelated local changes when switching branches.

For new V2 work, read `V2_RESEARCH_AND_BUILD_PLAN.md` and the relevant references in `docs/v2/`. The plan is a gated architecture, not proof of latency, quality or model fit. `docs/v2/V1_AUDIT.md` describes the actual legacy implementation and known failures. Preserve V1 unless the current task explicitly changes it; keep V2 dependencies and runtime separate.

**Implementation entry point:** read `docs/v2/IMPLEMENTATION_PHASES.md` and `IMPLEMENTATION_STATUS.md`. Start P00 when implementation is requested; resume the current phase on “continue.” Implement the requested phase, fix relevant failures, and update status with actual commands/evidence and the next task. Do not repeat the broad research survey or fabricate a completed gate. Planned paths/CLI names are not implemented features.

**User-selected TTS is Qwen3-TTS.** Begin with `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, X-Vector first; ICL is an explicit option. Do not make a Pocket/Kokoro/Fish comparison or independent winner declaration a prerequisite to implementation. P02 qualifies real PCM streaming, reference parity, cancellation and memory. Other TTS families are deferred; no silent substitution.

This is a **research project**. Select models using independent evaluations and actual user evidence, including noncommercial models on equal quality/technical criteria. `docs/v2/MODEL_QUALITY_BENCHMARKS.md` defines the current shortlist and Q0 gate. Author benchmarks and author-run user studies cannot establish an independent winner. Do not invent usage scores from stars, downloads or issue counts. Record licenses now; defer commercial-distribution clearance to that later profile. Actual research access requirements and consent still apply.

Complete delivery is defined by `docs/v2/E2E_IMPLEMENTATION_SPEC.md`, `DEPLOYMENT_AND_RELEASE.md` and `IMPLEMENTATION_BACKLOG.md`: real browser conversation, enrollment/consent, durable APIs, authenticated deployment, revocation, recovery and release evidence. The vertical slice is a gate, not the final product. Do not claim completion from mocked providers or an offline video demo.

The initial milestone is one consented avatar and one voice: known streamed text → PCM → motion → rendering → actual playback. Prove timing, cancellation and retargeted visual quality before adding the full conversation loop or expensive training. No model download, training run or cloud deployment is implied merely by reading this plan.

## Architecture invariants

- Providers are independently replaceable ASR, LLM, TTS and motion adapters with capability contracts. Verify actual incremental output; a batch result sliced into chunks is not low-latency streaming.
- One session actor owns transcript revisions, turn state, generation epochs and presentation acknowledgments. Share immutable model weights; isolate mutable session and voice state.
- Audio presentation is the master clock. Use integer sample ranges, explicit rate/timebase mappings, stateful resampling, EOF and discontinuity events.
- Carry session/turn/epoch/PTS through all stages. Discard stale output after cancellation, including buffered playback. Never silently drop speech to reduce queue length.
- Motion and renderer cadences are independent. Protect speech articulation from optional expressive motion; interpolate only within a valid epoch/basis.
- Use self-hostable LiveKit transport; the native candidate is C++ pre-encoded publishing with external NVENC, GStreamer fallback. Qualify the exact SDK/runtime and handle keyframe/rate feedback.
- Sender queue clearing does not flush receiver buffers. Browser cancellation requires a proven safe resume boundary; track replacement is the conservative fallback.
- Durable PostgreSQL/API/job state is separate from live PCM/frame queues. Session leases use fencing; asset revocation wins concurrent enrollment/publication.
- One media owner manages GPU resources and surface lifetimes per device. Explicit stream/event dependencies must cover render, conversion and encode consumers.
- Bound queues by time as well as item count. Admission protects active sessions; peak FPS alone is not capacity.
- Rig/Gaussians is the integration baseline, not a proven visual winner. Research GaussianAvatars/VHAP/LAM/EmoTaG components are eligible; a clean rewrite is not required before quality validation. A2F Claire has 260 ms future context; use StreamingTalker as a matched-rig challenger if the baseline fails, not a prerequisite to the first slice.
- Qwen3-TTS is the default GPU speech provider. Keep ASR/turn detection on CPU and initially use CPU LLM inference or an explicitly configured remote LLM. Qualify Qwen+motion+render together before GPU LLM co-residency. LiveAvatar is a deferred server visual reference, FlashHead Lite a conditional challenger; no assumption that every model fits on 16 GB.

## Assets, rights and compatibility

Keep source assets/checkpoints immutable. Every serving asset needs model/basis/topology hashes, camera/pose conventions, provenance and allowed uses. Reject silent coefficient padding, vertex truncation and face-index clamping. Equal dimensions do not prove compatibility.

Old FLAME/VHAP/GaussianAvatars and pretrained LAM remain eligible research components; their terms do not automatically permit commercial distribution. FLAME2023Open does not retroactively relicense old checkpoints. Record code, weight, dataset, voice, texture and evaluator terms separately without downgrading a model's research quality score. Do not apply a root license to unrelated existing work. Filenames or public availability do not establish consent.

Do not load untrusted pickle, arbitrary remote model code or unknown engine binaries. Keep biometric media out of routine logs and public benchmark artifacts. Scope caches and asset authorization by tenant.

## Development and validation

Use the proposed `v2/` layout when implementation begins; it does not yet exist. Prefer typed contracts and explicit configuration over global paths/device assumptions. Keep CPU contract tests independent of model downloads. Pin dependencies, source revisions, model hashes and runtime images when provisioning a real environment; do not invent a lockfile from untested versions.

Read `docs/v2/VALIDATION_PROTOCOL.md` for benchmark/quality changes. Distinguish independent human evaluation, independent reproduction, practitioner experience, author report, local observation, target and projection. Q0 qualifies quality/runtime parity and expensive training/release choices; it does not reopen the user-selected Qwen family or block foundation/functional integration while raters are unavailable. Scores from different protocols, stream modes, voices or endpoint/checkpoint versions are not interchangeable. Measure response from the annotated client speech endpoint to meaningful audio plus aligned video presentation. Report false cutoffs, failures, lookahead and cold/warm conditions. Do not sum stage p95 values and claim a measured system p95.

Observed baseline environment: system Python 3.14 had the audit dependencies; `.venv` lacked torch. This is not a reproducible V2 environment. The full V1 test suite currently fails collection for missing dependencies; do not claim it passes or install globally to hide that finding.

Useful existing checks from the repository root:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m pytest -q \
  tests/test_audio_driver_motion.py tests/test_runtime_animation.py \
  tests/test_tts_audio_bridge.py tests/test_orchestrator_stt.py \
  tests/test_orchestrator_transcript.py tests/test_tts_emotion_mapping.py \
  tests/test_virtual_camera_output.py
```

For an explicitly requested cached-model/GPU audit:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 \
  docs/v2/benchmarks/audit_probe.py --output /tmp/avatar-audit.json
```

Choose validation appropriate to the changed boundary. The probe measures isolated synthetic operations, not avatar quality or conversational latency. Report environment blockers candidly. PRs should state behavior changed, validation evidence, model/data/license impact and remaining limitations.

## Repo-local procedures

Focused skills in `.agents/skills/` cover research verification, real-time benchmarking, A/V synchronization, GPU media analysis, complete E2E delivery and deployment/release verification. Load only the relevant skill. `docs/v2/AGENT_ENVIRONMENT.md` documents discovery, available tools and environment conventions. Preserve unrelated user changes and global Codex settings.
