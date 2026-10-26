# V2 implementation status

Updated 2026-10-26. **P00 foundation and P01 CPU clock/replay/cancellation are implemented and validated.** This file records actual progress, not planned completion. Follow [implementation phases](IMPLEMENTATION_PHASES.md) and the [main plan](../../V2_RESEARCH_AND_BUILD_PLAN.md).

## Current checkpoint

- **Next action:** review the private `v2/var/qualification/listening-matched` clips, resolve the possible repeated word in the native long case, and qualify an authorized human reference before human-cloning acceptance. First-request horizon remains 259 ms against the 240 ms target. No further model downloads. Preserve the CPU-only contract environment.
- **Selected TTS:** Qwen3-TTS, initially `Qwen/Qwen3-TTS-12Hz-1.7B-Base`; X-Vector first, ICL an explicit option. Runtime and model revision/hash must be qualified in P02.
- **Phase in progress:** P02. Functional streaming, ordered segments/EOF, 30-minute sequential stability and real local Qwen playback/interruption pass on the graph-disabled, serial-decode profile. Human listening/cloning acceptance and the first-request latency shortfall remain open. Graph/async failures remain recorded.
- **Implementation dependencies:** selected model/tokenizer artifacts and a labeled synthetic reference are provisioned. An authorized human reference and independent listening are unavailable. P03 also needs a compatible authorized avatar. Functional work proceeds without representing synthetic inputs as those missing qualifications.
- **Preserve:** V1 source/checkpoints, pre-existing `data/README.md` and `output/README.md` deletions, and `.git-empty-master-backup/`. Reinspect working-tree state at implementation start; do not assume this inventory remains exhaustive.

## Phase ledger

| Phase | Status | Evidence / next action |
|---|---|---|
| P00 | complete | Python 3.12 editable install + actual lock; help/doctor; 37 offline CPU checks; fresh offline runtime-only install; [baseline evidence](../../v2/docs/baseline.md) |
| P01 | complete | Typed contracts, sample ledger, reference FIR, bounded queues, epoch cancellation and executable synthetic replay; 121 offline tests; [P01 evidence](../../v2/docs/p01.md) |
| P02 | in_progress | 430 requests / 1,813 s / zero failures; real early local playback and cancel/resume; 173 offline CPU checks; human quality/reference pending and fresh-cohort horizon 259 ms exceeds 240 ms target; [P02 evidence](../../v2/docs/p02.md) |
| P03 | not_started | Real avatar, Qwen/motion/render co-load and Q0 clips |
| P04 | not_started | Browser conversation and interruption |
| P05 | not_started | Enrollment, review, consent and revocation |
| P06 | not_started | Native GPU media, packaging and virtual devices |
| P07 | not_started | Admission, capacity, telemetry and soak |
| P08 | not_started | Security, deployment and release recovery evidence |
| P09 | deferred_optional | Client-rendered tier |

## Update contract for every implementation session

Use `not_started`, `in_progress`, `blocked`, `complete`, or `deferred_optional`. A blocker is specific missing input/failing dependency; list independent work still possible. Partial implementation stays `in_progress`; do not mark complete from code existence alone.

Append a dated session entry containing: phase/tasks completed; source/config paths; exact check/demo commands and outcomes; artifact paths with model/runtime/fixture hashes where applicable; unmet acceptance criteria; and the next executable task. Separate functional completion, performance acceptance and human-quality acceptance. Record scope changes/ADR decisions and unresolved inputs. Do not put secrets, raw voices, likeness media or personal identifiers here.

## 2026-10-26 — P00 complete

Implemented only the foundation: [package/config/build metadata](../../v2/pyproject.toml), [resolved application/dev lock](../../v2/uv.lock), `v2/src/talkinghead_v2/` (strict typed TOML config, `tha` CLI, bounded read-only diagnostics), `v2/tests/`, [local doctor config](../../v2/configs/local/doctor.toml), [setup instructions](../../v2/README.md) and [baseline/legacy inventory](../../v2/docs/baseline.md). No browser/native/provider/database scaffolding was added. P00 covers minimal E00/E01; the remaining E00/E01 deliverables and P01–P08 are not complete.

Actual environment: CPython 3.12.13, uv 0.10.9, isolated **`v2/.venv`**, editable `talkinghead-v2` 0.1.0; system-site packages disabled. Core runtime requirements are empty. Pytest 8.4.2 and resolved dev dependencies are pinned in `uv.lock`; hatchling 1.27.0 is the tested build backend. Lock SHA-256: `e392747f1cc84b27da55319c712e6077b7e9f9f53a5af0e8e8e40aa48a27f9d6`. No model/runtime/voice/avatar/biometric fixture was provisioned; inference pins await P02's actual compatibility qualification.

Commands executed from `v2/`:

```sh
uv sync --python /usr/bin/python3.12 --default-index https://pypi.org/simple
uv sync --locked --offline --python /usr/bin/python3.12
.venv/bin/tha --help
.venv/bin/tha doctor --profile local --config configs/local/doctor.toml
.venv/bin/tha doctor --profile local --config configs/local/doctor.toml --json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q
```

All returned 0; **37 passed in 0.21 s**. A fresh temporary runtime-only environment was reconstructed with `uv sync --locked --offline --no-dev --python /usr/bin/python3.12` and `UV_PROJECT_ENVIRONMENT` pointing outside the repository; installed help/imports and editable metadata passed with no runtime requirements. The exact temporary-environment command is retained in the baseline. Import smoke blocks inference imports, network connects, subprocess launches and dynamic-library loads. Config/failure tests run without GPU tools or model access. This is functional foundation evidence, not performance/human-quality acceptance.

Doctor reported Python/environment OK, RTX 4070 Ti SUPER driver access and FFmpeg compiled NVENC entries; optional `nvcc` and the configured V2 artifact cache were absent (warnings). Output permission/free-space preflight passed without creating directories or writing a probe. CUDA execution, usable encoding and model/cache validity remain unqualified. Final source/document whitespace and local-link checks passed; `git diff --check` passed.

V1 source/checkpoints, the root V1 environment and unrelated changes were preserved. Known strict checkpoint-load failures and full V1 test collection blockers are recorded from the existing audit in the baseline; no V1 test/model rerun or global dependency install was used to conceal them. No model/data/license/consent change or external deployment occurred.

**Unmet P00 acceptance:** none. **Next:** P01 only; preserve the no-model CPU boundary while implementing clock/replay/cancellation contracts. Real Qwen streaming/playback, retargeted avatar quality, browser conversation, independent ratings and release evidence remain later phase gates.

## 2026-10-26 — P01 complete

Added `v2/src/talkinghead_v2/contracts/`, `audio/{clock,resample,queue,motion}.py`, `session/{epochs,synthetic}.py`, the packaged procedural `fixtures/p01.json`, and `tha replay`. [P01 notes](../../v2/docs/p01.md) document contracts, reproduction, limitations and the offline wheel-install check. [README](../../v2/README.md) includes the runnable command. New CPU tests cover faults and boundary behavior; the P00 dependency lock/environment remain unchanged.

Native and canonical sample ledgers reject gaps/overlaps, reorder/duplicates and foreign generations/rates. The rational FIR preserves chunk-independent samples and EOF tails with an exact fractional valid duration. Queues bound duration and items without eviction. Cancellation fences ingestion, blocked producers/consumers and already-dequeued sink callbacks; task cancellation or a processing failure cancels the whole generation explicitly. EOF and discontinuity/cancellation follow separate paths. Motion interpolation rejects incompatible basis/topology/ordering/timebase or generation identity. Source segment labels survive filter lookahead.

Commands executed from `v2/` (all returned 0):

```sh
uv sync --locked --offline --python /usr/bin/python3.12
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/tha replay --report benchmarks/reports/p01-replay-2026-10-26.json
.venv/bin/python -O -m talkinghead_v2 replay
.venv/bin/tha --help
```

**121 passed in 0.76 s**. This includes all 16 pairings of 16/24/44.1/48 kHz, irregular/empty chunks, one-sample/partial tails, independent direct FIR oracle and filter checks, exact timebase conversion, both queue limits, blocked EOF/producer cancellation, delayed/dequeued stale callbacks, discontinuity/restart, injected processing failure and motion incompatibility. Imports remain free of inference/network/CUDA initialization. A wheel built with `uv build --offline --wheel` installed into a temporary Python 3.12 environment and ran the packaged replay outside the repository; the exact command is in the P01 notes. No runtime dependency was introduced.

Stored [synthetic replay evidence](../../v2/benchmarks/reports/p01-replay-2026-10-26.json): 1,800 frames at 24 kHz in the cancelled epoch; 2,222 frames at 44.1 kHz → **2,419 frames at 48 kHz** after restart, including **35 EOF-tail frames** and one EOF. Exact valid end is 355520/147 output units. Six stale callbacks were rejected. Four backpressure waits; peak 974 frames / three items within 1,440-frame / three-item bounds. Scheduled session PTS remained continuous. Fixture SHA-256: `c2f4011018597185e1ddcd801d018942dea73b5635393e6d3f1c7610ce921d19`; completed PCM SHA-256: `81038acc4264c91c4079ef04417f7487e6207e0ffd5fe25b3ccff5a38e0c7833`. The report includes the current package source digest, unchanged lock digest and sample/PTS trace; no raw human media or invented model/runtime hash is present.

**Acceptance boundary:** P01 functional CPU contracts passed; no unmet P01 criterion. The sink is explicitly a synthetic scheduling recorder. It does not acknowledge device/receiver playout, provider compute cancellation, RTP/drift recovery, real-time resampler throughput, sub-second conversation or human quality. Full M1A and later E2E/release gates remain open; no real-device/media claim is inferred from these tests.

V1 source/checkpoints/root environment and unrelated working-tree changes were preserved. No model download, global install, training, license/consent/data change, service or cloud deployment occurred. Source/document whitespace, JSON/report identity and local-link checks passed; `git diff --check` passed. **Next: P02**, with actual Qwen/cache/runtime/reference inputs still requiring explicit qualification.

## 2026-10-26 — P02 provisioning/build checkpoint (in progress)

The user requested P02 and GitHub checkpoints. The completed P00/P01 implementation and its planning context were committed as `480928a` and pushed to `origin/v2/p02-qwen-streaming`; main was preserved. Existing data/output README deletions and the untracked Git backup remain outside commits.

Added explicit `tha artifacts provision|verify`, a selected-Base-only manifest, size/SHA-256 checks and an immutable content-addressed cache with read-only snapshot verification. Startup verification never fetches missing files. Voice registry validation requires current approval, allowed uses, tenant identity and audio hash. No human reference was supplied; the local reference is explicitly an eSpeak-generated `synthetic_fixture`, approved only for plumbing. It is private under ignored `v2/var/`, not a claim of consented human cloning quality.

Actual model pin: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` at `fd4b254389122332181a7c3db7f27e918eec64e3`; eleven files, **4,544,170,364 bytes**, provisioned and verified under `v2/var/artifacts`. Bundled tokenizer weights match `Qwen/Qwen3-TTS-Tokenizer-12Hz` revision `7dd38ad4e9bad454aae9cd937d0cd577604fe229` (hashes in the manifest). No model files are committed. Model/tokenizer card terms are recorded as Apache-2.0 separately from voice/tool terms.

Pinned official source `022e286b98fbec7e1e916cb940cdf532cd9f488e`; isolated `v2/runtimes/qwen-reference/.venv` resolves torch/torchaudio 2.10.0+cu128 and the official requirements. Its lock is separate from the CPU foundation. An official offline adapter uses local safetensors, bf16/SDPA and X-Vector first, with ICL an explicit transcript-requiring option. Initial real reference generation is running; no streaming claim follows from that batch API.

Native source `maxhorizons/qwen3-tts-cpp-streaming` is pinned to `74422bab4998ad69e0a67ba105654dadbeec9b3a`; bundled GGML tree is `b1e00d0467ff3e8a7ade3bc6031b756b4cfe0c13`. CUDA 12.8.93 was present outside PATH. Initial configuration failed at CUDA/glibc math exception declarations; a private six-declaration header overlay fixed that boundary without modifying system files. Compilation then exposed three Windows-only player metrics referenced on Linux; the narrow checked-in patch returns unavailable metrics on Linux. With GCC 14.3.1, architecture 89 and CUDA graphs enabled, the native CLI and wrapper contract test built; `qwen3_wrapper_contract_test` reported PASS. Base 1.7B and tokenizer safetensors converted to F16 GGUF locally with zero TTS tensors skipped. Derived model hashes and measured streaming evidence will be recorded after runtime qualification.

Commands already passed: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q` from `v2/` — **145 passed in 0.80 s**, including corruption/truncation, unsafe paths, missing tokenizer, floating revision, wrong provider family, symlink/extra-file rejection, offline startup and cross-tenant/revoked/unconsented voice denial. Explicit model provisioning and offline verification passed. Native build logs, conversion outputs, reference audio and embeddings remain in ignored private `v2/var/`.

**Unmet acceptance:** real incremental callback adapter and ordered segment lifecycle; audible output before synthesis completes; compute-cancel acknowledgment and safe local resume; first 100/300/500 ms horizons, production gaps, RTF and memory; 30-minute sequential stability; official/runtime listening/WER and authorized human cloning parity. P02 remains **in_progress**. Next: complete the native pipe adapter and local audio sink, then run these real-model checks. No alternative TTS family, training, global install or cloud deployment was introduced.

## 2026-10-26 — P02 streaming/device checkpoint (in progress)

Provisioning/reference checkpoint `2bd1c82` was committed and pushed to the same feature branch. The official offline adapter completed all five real-model text cases, with bf16/SDPA and the synthetic reference: observed RTF approximately 0.73–0.78, warm-filesystem model load 1.02 s and prompt construction 0.68 s. The actual embedding dimension is 2,048. [Initial reference evidence](../../v2/benchmarks/reports/p02-official-reference-initial.json). The final review below records its sampling mismatch and a separate corrected run.

Implemented the persistent native pipe adapter (`open_session`, ordered text `push`, async audio, flush, compute cancel and close), request serial/epoch fencing, EOF/safety-limit distinction, bounded callback backpressure and explicit worker-termination recovery. The first five native graph/async cases generated real early PCM: RTF 0.31–0.53; first PCM and 300 ms production horizon 238–296 ms; sampled whole-device peak about 5.1 GiB including desktop usage. These measurements belong to that experimental profile; its eventual 240 ms horizon-p95 target was not passed. [Initial cohort](../../v2/benchmarks/reports/p02-native-cohort-graphs-async.json).

Two reliability failures were preserved. The first run was interrupted after 245 successful requests and one invalid protocol frame (epoch 184); only intermediate aggregate/last-request data survived. The bridge was hardened with a dedicated PCM descriptor and library stdout redirected to diagnostics. A compiled CPU fault regression now injects upstream stdout across 100 requests. The second run retained complete per-request observations and failed at epoch 262 with cuBLAS unsupported-parameter and illegal-memory-access errors, then recovered through an explicitly counted worker restart. It was stopped at 947.44 s with 309 successful requests and one failure. Neither is a stability pass. [Initial failure](../../v2/benchmarks/reports/p02-initial-soak-failure.json), [graph/async trace](../../v2/benchmarks/reports/p02-native-soak-graphs-async.json).

A conservative **same Base 1.7B/F16/X-Vector** profile now disables CUDA graphs and asynchronous decode through explicit per-worker flags. It is running a fresh 30-minute test; no success is claimed in advance. Source/build reproduction, GPU header overlay, runtime pinning, isolated model conversion and benchmark commands are documented in [P02 notes](../../v2/docs/p02.md). The core still has no inference dependency.

Added a typed sink boundary, P01 local-software cancellation scope, bounded callback-paced PortAudio output and output-monitor capture. Real tone calibration passed on the current local output: both old/new tones detected, no old 440 Hz energy after resumed 880 Hz plus the 50 ms analysis boundary, 25 ms observed transition gap. No microphone was captured; this is output-monitor evidence, not acoustic/human quality or an exact wall-clock suppression bound. Actual Qwen speech/device and interrupt/resume commands are implemented and queued after the single-worker soak. [Calibration](../../v2/benchmarks/reports/p02-device-calibration.json).

CPU-only strict ASR evaluation of the initial same-text/voice clips used pinned Whisper Base English safetensors as an evaluator, not a serving-ASR choice. Sampling controls differed, as identified in the final review below. Official reference: 12/150 word errors (8%); native graph/async: 15/150 (10%). Number/abbreviation/accent alternatives and ASR mistakes count under the declared normalization. A private blinded listening packet exists, with no fabricated ratings. [Evaluator manifest](../../v2/configs/quality/whisper-base.manifest.json), [rows/limits](../../v2/benchmarks/reports/p02-intelligibility-graphs-async.json).

Latest offline CPU suite: **158 passed in 1.61 s**; CUDA build/wrapper contract, Base/tokenizer conversions and device calibration also passed at their declared boundaries. No models, embeddings, raw speech, capture media or build trees are committed. **Unmet:** stable 30-minute serving profile, real Qwen playback/cancellation evidence, the low-latency target, independent listening and authorized human clone parity. P02 remains **in_progress**, with actual failures guiding the same-model runtime configuration.

## 2026-10-26 — P02 functional qualification and final evidence

Streaming/device checkpoint `a97cb54` was committed and pushed to `origin/v2/p02-qwen-streaming`. The final serving profile retains **Base 1.7B/F16/X-Vector**, with per-worker `GGML_CUDA_DISABLE_GRAPHS=1` and `THA_QWEN_ASYNC_DECODE=0`. Its [sequential soak](../../v2/benchmarks/reports/p02-native-soak.json) completed **430 requests in 1,813.235 s with zero failures**, no interruption and no stale native packets. RTF p50/p95 was **0.461/0.592**, 300 ms production-horizon p95 **223.6 ms**, queue peak 19,200/24,000 samples, and sampled whole-GPU peak **5,508,169,728 bytes (5.13 GiB)** including desktop usage. This is one ready worker repeating five synthetic-reference cases; it does not measure conversational presentation, mixed-worker residency or admission capacity.

The separate [fresh five-case cohort](../../v2/benchmarks/reports/p02-native-cohort.json) recorded RTF p50/p95 **0.460/0.587** and 300 ms horizon p95 **259.1 ms**, above the eventual 240 ms target. The first request was after model/voice readiness; this is not cold-install latency. The warm-soak percentile does not establish a universal latency pass. Per-request first 100/300/500 ms horizons, maximum production gaps and callback traces are retained, including failed graph/async profiles.

[Real Qwen playback](../../v2/benchmarks/reports/p02-local-speech.json) connected the native adapter through P01's sample ledger/resampler/epoch queue to PortAudio. **166,485 samples at 24 kHz → 332,970 valid samples at 48 kHz**, including final tail; output-monitor audio arrived before generation returned. [Interruption/resume](../../v2/benchmarks/reports/p02-local-interruption.json) fenced epoch 0, acknowledged cooperative compute cancellation in **146.6 ms**, aborted/closed the old stream, then delivered epoch 2's **120,405 → 240,810 samples**. The final 50 ms requested device latency / 960-frame callback profile had zero software underruns and zero PortAudio status callbacks in normal, interrupted and calibration runs. Initial lower-latency runs had status warnings and are retained privately. [Final tone calibration](../../v2/benchmarks/reports/p02-device-calibration.json) detected both tones with no old tone after the new tone plus a 50 ms analysis boundary; transition gap 75 ms at 25 ms analysis resolution. Monitor capture is not acoustic listening or a measured wall-clock/browser suppression bound.

The [three-segment fixture](../../v2/benchmarks/fixtures/p02-segments.json) generated [53,205 + 58,965 + 58,965 = 171,135 samples](../../v2/benchmarks/reports/p02-segments.json), with contiguous segment labels and exactly one EOF. CPU ASR transcribed all three sentences and the final word; “twelve point five” appeared as “12.5” and counts as three strict word errors. Empty, oversized and malformed segment inputs now fail before worker startup. Device-start failure closes the partial stream; explicit synthetic I/O tests cannot claim actual device playback.

Final review found that the initial official reference inherited residual sampler defaults and used a different main top-p from native. Those historical comparisons are retained with that limitation. A [new offline reference](../../v2/benchmarks/reports/p02-official-reference.json) explicitly matches native controls: main/residual temperature 0.75 and top-k 16, codebook-0 top-p 1.0, residual top-p 0.9, repetition penalty 1.02, ceiling 1,024 and seed 17. RNG implementations, per-request versus per-process seed initialization and bf16/F16 numerics still differ. New reference load/prompt times were 0.917/0.404 s; RTF 0.74–0.81. Its 2,048-dimensional embedding SHA-256 remains `70c6e7218826851d99c7ff40e6b56963a2bf1232f1d062d501be5058122f3893`, identical to native conditioning.

[Corrected strict ASR comparison](../../v2/benchmarks/reports/p02-intelligibility.json) remains **official 12/150 errors (8%), native 15/150 (10%)**. These are local ASR observations, not independent human scores or demonstrated runtime quality parity. The native long case includes a transcribed repeated “boundary,” pending listening review. The final private blinded/gain-matched packet is `v2/var/qualification/listening-matched`; no rating was supplied or invented. No authorized human voice reference was supplied. P02 therefore remains **in_progress**, despite passing its functional/device/stability checks.

Actual commands from `v2/` include the following; exact model/device/text/quality commands are retained in [P02 reproduction notes](../../v2/docs/p02.md):

```sh
uv sync --locked --offline --python /usr/bin/python3.12
uv sync --project runtimes/qwen-reference --locked --offline --python /usr/bin/python3.12
uv pip check --python .venv/bin/python
uv pip check --python runtimes/qwen-reference/.venv/bin/python
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q
.venv/bin/python scripts/build_qwen.py --cuda-root /usr/local/cuda-12.8 --cxx /usr/bin/g++-14 --jobs 4
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/tha tts benchmark \
  --runtime var/native-runtime-serial.json --texts configs/qwen/reference-texts.json --seconds 1800 \
  --report var/qualification/native-soak-serial.json --log var/build/native-soak-serial.log
.venv/bin/python scripts/publish_qwen_evidence.py
```

All returned 0. **173 offline CPU tests passed in 3.43 s.** Both dependency checks passed; a fresh wheel installed outside the repository ran help/replay without torch or sounddevice. The full build helper reproduced the qualified bridge SHA-256 `82e7af68318dc15a840ecb702a91d5d85922381db1d73eb689b3d9b45d11d6fe`; model/converter/runtime identities and private-overlay hashes are in [build evidence](../../v2/benchmarks/reports/p02-native-build.json). Native adapter/binary/config were unchanged after the successful soak; device/report/input-validation changes were checked separately. The original CPU lock is unchanged. Core imports initialize no inference runtime, network or audio device. The full V1 suite was not rerun; its known collection blockers remain, and V1 source/root environment were preserved.

**Constraint deviation:** earlier P02 provisioning downloaded model/evaluator weights despite the user's original no-download instruction. This exceeded that constraint. Final qualification used those local caches offline and made no further weight downloads. No weight, embedding, raw speech, capture media, build tree or private registry is committed. No training, global package/driver change, cloud deployment or TTS-family substitution occurred. Model, runtime, evaluator and synthetic-tool terms remain separately recorded; human consent and research/release quality are not inferred from synthetic assets.

**Next executable task:** review `listening-matched/packet.json` without its key, record missing/repeated words, clicks, tail completion and intelligibility/naturalness; investigate any confirmed native/reference regression. Qualify a consented, authorized human reference when available. Preserve the fresh-request horizon shortfall rather than claiming an unrestricted performance pass. P03's actual compatible avatar, retargeting and Qwen/motion/render co-load remain unstarted.
