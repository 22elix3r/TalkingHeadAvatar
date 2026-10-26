# V2 implementation status

Updated 2026-10-27. **P00 foundation and P01 CPU clock/replay/cancellation are implemented and validated.** This file records actual progress, not planned completion. Follow [implementation phases](IMPLEMENTATION_PHASES.md) and the [main plan](../../V2_RESEARCH_AND_BUILD_PLAN.md).

## Current checkpoint

- **Next action:** exercise the local browser with an actual user and the declared capture/output profile, qualify pause/echo/cutoff behavior, and improve the measured 3.41 s client-graph reference latency without hiding endpoint patience. Identify cached Claire and an authorized avatar/calibration to close P03 and P04's speaking-video gate. P02 listening/reference and fresh-cohort latency findings remain open. No further model downloads; preserve the CPU-only core.
- **Selected TTS:** Qwen3-TTS, initially `Qwen/Qwen3-TTS-12Hz-1.7B-Base`; X-Vector first, ICL an explicit option. Runtime and model revision/hash must be qualified in P02.
- **Phase in progress:** P04, explicitly requested as the next phase. The real local browser audio reference passes controlled synthetic-input functional/recovery checks. P03's real speaking avatar and P04's meaningful acoustic/aligned-video/human/latency gates remain open. Advancing independent browser work does not close P02/P03 findings.
- **Implementation dependencies:** selected model/tokenizer artifacts and a labeled synthetic reference are provisioned. An authorized human reference and independent listening are unavailable. P03 also needs a compatible authorized avatar. Functional work proceeds without representing synthetic inputs as those missing qualifications.
- **Preserve:** V1 source/checkpoints, pre-existing `data/README.md` and `output/README.md` deletions, and `.git-empty-master-backup/`. Reinspect working-tree state at implementation start; do not assume this inventory remains exhaustive.

## Phase ledger

| Phase | Status | Evidence / next action |
|---|---|---|
| P00 | complete | Python 3.12 editable install + actual lock; help/doctor; 37 offline CPU checks; fresh offline runtime-only install; [baseline evidence](../../v2/docs/baseline.md) |
| P01 | complete | Typed contracts, sample ledger, reference FIR, bounded queues, epoch cancellation and executable synthetic replay; 121 offline tests; [P01 evidence](../../v2/docs/p01.md) |
| P02 | in_progress | 430 requests / 1,813 s / zero failures; real early local playback and cancel/resume; 173 offline CPU checks; human quality/reference pending and fresh-cohort horizon 259 ms exceeds 240 ms target; [P02 evidence](../../v2/docs/p02.md) |
| P03 | in_progress | 206 offline CPU checks; strict mesh import, bounded Claire context frontend, explicit retarget mathematics, cached-Qwen/procedural mesh file replay; actual Claire/authorized avatar, presented A/V, co-load and Q0 missing; [P03 evidence](../../v2/docs/p03.md) |
| P04 | in_progress | Firefox 149/157 and Chromium local audio/recovery; 213 core + 7 PostgreSQL/fault + 2 web-unit + 8 Chromium + 1 Firefox checks, installed-Firefox failure/retry/playback flow; latest n=1 graph proxy 4.92 s; real avatar/A-V/human gates open; [P04 evidence](../../v2/docs/p04.md) |
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

## 2026-10-26 — P03 timing and compatibility checkpoint

The user explicitly requested the next phase after P02 checkpoint `ac8c9c8`. Work proceeds on `v2/p03-avatar-timing`, preserving P02's open quality and latency findings. No additional model weights, engine binaries or datasets were downloaded. No training, cloud deployment, global package/toolkit change or V1 source change occurred. Unrelated README deletions and the Git backup remain untouched.

Implemented [canonical mesh loading](../../v2/src/talkinghead_v2/assets/avatar.py), [Claire audio context preparation](../../v2/src/talkinghead_v2/providers/motion/claire.py), [explicit retarget/landmark calibration](../../v2/src/talkinghead_v2/retarget.py), [rigid-jaw mesh rendering](../../v2/src/talkinghead_v2/rendering/mesh.py), [bounded motion/video timelines](../../v2/src/talkinghead_v2/rendering/timeline.py) and [offline replay](../../v2/src/talkinghead_v2/avatar_replay.py). `tha avatar verify` rejects unsupported representations, unauthorized metadata, basis/topology mismatches, vertex/face coercion and invalid camera conventions. `tha avatar replay` runs a procedural mesh test pattern, with either calibration audio or provenance/hash-verified cached Qwen pilot PCM. It does not generate neural motion, run a real avatar or claim presented lip sync.

Source-only inspection pinned [Claire metadata and SDK identities](../../v2/configs/motion/claire-source.json): model metadata revision `a46eafb067adfdd7cc8f5c7941586b309004561c`, SDK commit `1ca0f02535ed774f5dbcd724a31cd486368dc783`. Only bounded text/source metadata was fetched; no fetched code was executed. The 8,320-sample/16 kHz window with 4,160 center offset establishes **260 ms future context**. The frontend separately accounts for P01 FIR right context and 20 ms chunk alignment. It bounds retained context to 8,640 samples, uses explicit EOF/valid duration, rejects gaps/reordering and fences cancellation without flushing playable model padding.

Scoped local discovery found no configured Claire cache/SDK engine in `v2/var`, the Hugging Face cache or usual local NVIDIA library locations; `ldconfig -p` listed no `libnvinfer`, `libnvonnxparser` or `libaudio2x`. The exact SDK runtime/build is therefore not qualified. Existing vendor sample PLY/NPZ and old FLAME files are present but have no current approval/consent/retarget metadata. No legacy loader or pickle was executed, no sample was treated as consented, and no topology-repair path was imported. A question requesting the actual cached paths/authorization was sent while independent work continued; no input was supplied by this checkpoint.

The packaged [eight-vertex mathematical mesh](../../v2/src/talkinghead_v2/fixtures/p03-avatar/avatar.json) contains no likeness or texture. Its explicit fixture controls exercise articulation-channel bounds, jaw rotation around a pivot, protected idle/listening behavior and the independent 50 Hz motion / 30 FPS renderer timeline. Calibrated mappings bind full source/target rig identities, so equal coefficient counts cannot bypass compatibility. Bounded landmark fitting is validated on known synthetic correspondences; it is not a fitted Claire-to-FLAME calibration. Gaussian integration remains unimplemented pending an actual compatible asset, rather than being represented by this mesh fallback.

[Calibration replay](../../v2/benchmarks/reports/p03-calibration.json) processed 72,000 samples at 24 kHz into 180 context windows and 90 RGB frames at 128². [Cached-Qwen replay](../../v2/benchmarks/reports/p03-qwen-short.json) processed **66,645 samples → 167 context windows and 84 video frames**, including 15 model-only EOF-padded windows. Peak context **8,619/8,640 samples**, motion history **3/64 items** with a 500 ms future bound, video mailbox **1/3 frames** with a 100 ms future bound. Completed stale callbacks were rejected after cancellation. The first window needed 6,720 native samples (280 ms generated audio) in this chunk/filter profile; mapping to P02's saved callback trace gives **259.15 ms after text submission**. That is a recorded input-readiness observation, not a new live neural/presentation latency measurement.

FFmpeg FFV1/PCM file mux/demux verification preserved all **66,645 Qwen float32 samples and 84 RGB frames** exactly; [container evidence](../../v2/benchmarks/reports/p03-qwen-short-container.json). The calibration container similarly preserved 72,000 samples / 90 frames. The mesh preview was visually inspected. These are offline file and sample checks, not an actual device A/V presentation test. Audio/RGB/PNG/container outputs remain private under ignored `v2/var/`; public evidence contains only non-biometric fixture metadata and synthetic-pilot metrics.

Commands from `v2/` (exact mux and reproduction commands in [P03 notes](../../v2/docs/p03.md)):

```sh
uv sync --locked --offline --python /usr/bin/python3.12
.venv/bin/tha avatar verify \
  --manifest src/talkinghead_v2/fixtures/p03-avatar/avatar.json --tenant procedural
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/tha avatar replay \
  --output var/qualification/p03-calibration --resolution 128
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 runtimes/qwen-reference/.venv/bin/tha avatar replay \
  --output var/qualification/p03-qwen-short --resolution 128 \
  --wave var/qualification/native-waves-serial/native-000.wav \
  --tts-report benchmarks/reports/p02-native-cohort.json --case short
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q
```

All returned 0; **206 offline CPU tests passed in 5.04 s**. New tests cover authorization/type/shape denial, actual mesh-pixel changes, rigid-jaw invariants, bounded regularized fitting, protected behavior, irregular/short/fractional EOF context, gap/rate/order rejection, both queue bounds, late video, monotonic playhead, completed stale callbacks and the runnable replay. Final review added explicit articulation/behavior channel authority and composition that preserves speech and rigid jaw pose; the final suite, replays and container checks include that guard. Core dependency checks and a fresh offline wheel/packaged replay outside the repository passed without torch or soundfile. The 30-minute **virtual-clock** stress test advances **108,000 motion frames** with at most two retained history items; it is not the required wall-clock playback soak. Replay CPU render timings overlapped other CPU qualification jobs and establish no FPS/performance gate. Report identities bind package source/diff, unchanged CPU lock, source WAV/benchmark, rig and inspected metadata.

**Acceptance still missing:** actual Claire model execution and cancellation; research-authorized real avatar with retargeted bilabials/teeth/identity/Q0 clips; timestamped Gaussian renderer; live Qwen through neural motion; measured local presented skew ≤40 ms and stale audio suppression ≤200 ms; 30-minute real A/V bounded-queue soak; measured co-loaded Qwen+motion+render GPU memory. **P03 remains `in_progress`; P04's functional-media prerequisite is not passed.**

**Next executable task:** use the requested local cached Claire paths and avatar authorization/basis/topology/camera records, verify immutable identities, build an isolated exact SDK runtime without weights downloads, fit and validate the real calibration, then qualify recorded and live Qwen playback with the actual renderer. Source-only SDK/model terms, procedural fixture terms and real-avatar rights are kept separate; no root license or human-quality score is invented.

## 2026-10-27 — P04 local browser audio reference

The user explicitly requested the next phase after `1c7ec02`; P04 independent browser work proceeded on `v2/p04-browser-reference` while P02/P03 gates remained open. Implemented [React device/playback/caption UI](../../v2/web/src/main.tsx), [FastAPI control plane](../../v2/src/talkinghead_v2/server/app.py), [PostgreSQL metadata/migration](../../v2/runtimes/session-api/migrations/versions/0001_sessions.py), [single session actor](../../v2/src/talkinghead_v2/server/actor.py), [CPU ASR/LLM adapters](../../v2/src/talkinghead_v2/server/providers.py), [LiveKit raw reference transport](../../v2/src/talkinghead_v2/server/transport.py), [receiver binding gate](../../v2/web/src/playback.ts) and [turn/segmentation policy](../../v2/src/talkinghead_v2/conversation.py). [Runbook](../../v2/docs/p04.md), [exported OpenAPI](../../v2/docs/p04-openapi.json), [qualification report](../../v2/benchmarks/reports/p04-browser-reference.json).

Real local path: Chromium microphone/typed input → CPU rolling-window ASR revisions and conservative VAD/turn commitment → actual streamed LLM text → bounded committed segments → the existing native Qwen PCM worker → P01 sample/resampling semantics → actual self-hosted LiveKit browser audio. Only the expected user's microphone is subscribed. Captions stay in bounded memory; durable tables store session/asset/idempotency/control metadata rather than raw PCM, prompts or transcripts. The UI explicitly uses a **synthetic voice and static procedural visual**. No real avatar, lip-sync or photorealistic claim is inferred from that static track.

[Source lock](../../v2/configs/p04/source-lock.json): cached `Systran/faster-whisper-base.en` revision `3d3d5dee26484f91867d81cb899cfcf72b96be6c`, copied/hashed/read-only, CTranslate2 CPU/int8 with two threads; cached Ollama **`qwen3:14b` Q4_K_M** digest `bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`, explicitly selected as the available CPU reference with 2,048-token context, four threads and `num_gpu=0`. Startup verified `size_vram=0`. This differs from the planned 4B LLM profile and has a measured latency cost; it is not a new model winner. TTS remains **Qwen3-TTS Base 1.7B/F16/X-Vector**, with P02 graph-disabled/serial flags and no family substitution. New Python/Web locks contain actually installed/tested dependencies; the core and P02 official-reference locks remain separate.

Created only the isolated `tha-v2-p04` Compose project, using pinned PostgreSQL and LiveKit image digests. API is fixed to `127.0.0.1:18700`; PostgreSQL and SFU published ports bind to loopback. Random development credentials/configuration remain ignored under `v2/var/p04/` with restricted permissions. No other existing database/container, global dependency, Ollama setting, V1 source or legacy environment was modified. No cloud deployment or training occurred.

Development identity enforces peer/Host/Origin/CSRF and bounded body checks. Durable session creation is tenant scoped and idempotent; a changed retry body conflicts. Atomic configured admission allows one preparing/active session. Join grants restrict room/identity/source and expire in two minutes. Epoch-checked controls reject duplicates with different content, stale ordering and cross-tenant access. Fencing rejects stale durable writes; native distributed lease enforcement remains unqualified. Lost browser heartbeats/unjoined sessions have a bounded 30-second grace plus lease poll. Concurrent close cleans sources once even when database persistence fails; startup fences/reconciles orphaned sessions.

The inspected SDK receive queue silently evicts on overflow. A pinned, guarded source-owned queue now reports an explicit audio discontinuity and fails the turn/session instead. Cancellation fences media before awaits, gates/detaches receiver output and rejects late old bindings even while HTTP cancellation is pending. New answer tracks have distinct IDs and epoch/clock mappings; server cancellation acknowledgment alone never unmutes. Network/control failure requires a new session; device change stops the old one. Physical speaker stale suppression ≤200 ms is still not measured.

Actual validation from `v2/`:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q --junitxml=var/p04/core-tests.xml
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 runtimes/session-api/.venv/bin/python \
  -m pytest -q runtimes/session-api/tests --junitxml=var/p04/runtime-tests.xml
uv sync --project runtimes/session-api --locked --offline --python /usr/bin/python3.12 --default-index https://pypi.org/simple
uv pip check --python runtimes/session-api/.venv/bin/python
```

**212 core tests passed in 5.15 s; six real-PostgreSQL/API/queue/fault tests passed in 1.13 s.** Providers are explicitly mocked in API boundary tests, with separate actual-model browser evidence. Migration downgrade/upgrade passed only against the dedicated `tha_p04_test` database. A fresh offline core wheel imported/replayed outside the repository with no FastAPI/torch dependency. CPU contract validation still needs no model access. The known full V1 collection failures were not concealed with global installs or a claimed V1 suite pass.

From `v2/web/`: `npm ci --offline --no-audit --no-fund`, `npm run build`, `npm run test`, and `./node_modules/.bin/playwright test --reporter=json > ../var/p04/playwright-final.json` passed. TypeScript/Vite production build passed; initial application JS is about 230 kB uncompressed and LiveKit is deferred until interaction. Vite still reports the deferred ~585 kB transport chunk warning. **Two receiver-gate unit checks and seven actual Chromium browser checks passed**, covering real typed/microphone conversation, replacement-track interruption, native permission denial, injected autoplay recovery, disconnect, device change, preparation cancellation and a controlled latency probe. The installed Chromium was used without fetching a browser binary. Synthetic test input came from locally generated eSpeak audio; it is not the serving TTS or a real human recording.

The initial sky-question fixture failed ASR both offline (“Cleodate Times guy”) and through the browser (“clear day times guy”). This is retained as a failure, not relabeled as a quality pass. A separately declared arithmetic plumbing fixture produced **“What is 2 plus 2?” → “2 plus 2 is 4.”** The harness captured received LiveKit Opus audio privately; independent CPU transcription of that capture confirmed **“2 plus 2 is 4.”**, including the final word. Screenshots were visually inspected. The source/lock/model/image hashes, fixture/receiver hashes, browser results and failure record are in the qualification report. Raw media/screenshots, keys and database files remain uncommitted.

The final controlled **client WebAudio graph proxy measured 3,406.1 ms, n=1**, from last nonzero microphone observation to first nonzero received-audio observation in one browser performance clock. Polling is 20 ms with an FFT window and uncalibrated graph/device/acoustic offset. First meaningful-word onset is not annotated; actual acoustic and aligned A/V presentation remain unqualified. No p50/p95/p99 or system SLO is inferred. The 1.2-second endpoint patience preserves the 200/500/1000 ms logic fixtures but exceeds the design endpoint budget; natural-human pause/echo/backchannel/cutoff rates still require a real cohort. The CPU 14B profile and conservative timing do not establish sub-second response.

**Constraint deviation:** the serving ASR/LLM/TTS checkpoints were already cached, but installing Faster-Whisper also downloaded its unused **1,245,151-byte `silero_vad_v6.onnx`** bundled in the dependency wheel. This exceeds a literal no-new-weight-files interpretation and is explicitly recorded with its hash. The bundled model was not loaded (`vad_filter=False`); serving VAD is WebRTC. No model-hub pull/provisioning or new serving checkpoint download occurred. Software dependency/SFU-image installation is recorded separately. No raw voice/model file or unrelated root license was published.

**P04 remains `in_progress`.** Still missing: P02 listening/reference quality; P03 authorized real avatar/Claire/calibration, speaking video, co-loaded memory and Q0; actual-user and echo/pause/interruption cohorts; calibrated meaningful acoustic endpoint→aligned presentation quantiles; real A/V soak and ≤200 ms speaker suppression evidence; hosted auth/security/TURN/release gates. **Next:** exercise the local UI with the intended user/device, fix the observed ASR/endpoint/CPU-LLM latency limits without changing timing boundaries, and identify cached authorized visual inputs before claiming the speaking-avatar media gate or advancing release acceptance.

GitHub correction: verified the destination as `https://github.com/22elix3r/TalkingHeadAvatar.git` and authenticated account as `22elix3r` (ID 157088510). Earlier commits used the incorrect author email `elix3r@users.noreply.github.com`. Repository-local attribution is now `22elix3r <157088510+22elix3r@users.noreply.github.com>`; global Git settings were not changed. Existing published commit history was not rewritten.

## 2026-10-27 — P04 Firefox connection and recovery correction

Resumed P04 on `v2/p04-browser-reference` after the user reported “could not establish pc connection” and identified same-device Firefox. The previous local SFU explicitly advertised `127.0.0.1`; Firefox normally rejects loopback ICE candidates. [Local service setup](../../v2/scripts/p04_local.py) now uses LiveKit's discovered private Docker bridge address with external-IP discovery disabled. Existing `up` migrates the exact old entry without replacing credentials. A configuration-hash service label lets Compose recreate only a changed SFU and safely retry an interrupted migration. An actual repeated `up` retained the same SFU container ID; the PostgreSQL container/volume was retained throughout. All Docker-published sockets, API, signaling and database remain bound to host loopback. This transport is qualified for a same-host browser with native Linux Docker, not Docker Desktop or a remote/forwarded browser.

[Client recovery](../../v2/web/src/client.ts) now translates SDK connection failures into a readable retry action and preserves cancellation's abort semantics. The failed-connection browser check verifies source/output cleanup, durable CLOSED status, released admission and successful retry. [API navigation](../../v2/src/talkinghead_v2/server/app.py) redirects a `localhost` root navigation to the configured preview origin; API alternate-Origin denial and CSRF remain enforced. The preview is running at **http://127.0.0.1:18700/** with the corrected SFU and rebuilt client. No user Firefox preferences/profile, global packages, network interfaces, firewall, V1 source or unrelated worktree changes were modified.

Actual validation from `v2/`:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q --junitxml=var/p04/core-firefox-fix.xml
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 runtimes/session-api/.venv/bin/python \
  -m pytest -q runtimes/session-api/tests --junitxml=var/p04/runtime-firefox-fix.xml
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/check_p04_firefox.py \
  --geckodriver var/p04/tools/geckodriver --output var/p04/firefox-system-final.json
```

**213 core tests passed in 5.20 s; seven PostgreSQL/API/fault tests passed in 1.46 s.** New coverage checks migration idempotence/credential preservation/custom-config preservation, canonical localhost navigation and continued Origin denial. From `v2/web/`, `npm run build` and `npm run test` passed; the deferred transport bundle still has the previously recorded size warning. `./node_modules/.bin/playwright test --reporter=json > ../var/p04/chromium-firefox-fix-final.json` passed **eight actual Chromium checks in 50.00 s**. `./node_modules/.bin/playwright test --config playwright.firefox.config.ts --reporter=json > ../var/p04/firefox-final.json` passed the real Qwen/LiveKit playback/interruption check in **Firefox 157, 7.18 s**, with loopback ICE explicitly disabled.

The separate stdlib WebDriver harness passed on the installed **Firefox 149.0.2**, also with loopback ICE disabled. A candidate-rewrite fault recreated the old loopback-only advertisement without restarting the SFU, produced the readable connection failure, and allowed retry. The normal private-address path received two generations of actual Qwen audio, gated interruption, replaced the old audio track/epoch, published a synthetic microphone and ended/restarted sessions. The selected media path was UDP to a non-loopback SFU candidate. Microphone publication alone is not a human-ASR pass; the Chromium arithmetic case separately exercised actual ASR and transcription of the received answer preserved **“2 plus 2 is 4.”** Physical speaker output and stale-speech timing remain unmeasured.

An initial attempt to drive installed Firefox using Playwright's `moz-firefox` BiDi channel timed out, including after the configuration change. Those failed harness runs remain private and are not counted as passes or used to establish final fault attribution. The successful installed-browser run used official geckodriver 0.37.1, with its release archive checked against SHA256 `e815130ea95983e162ae91843b48d3a3ce991735635fce83a647afde21e09f7e` and its extracted binary hash recorded. It resides only under ignored `var/p04/tools/` and uses a disposable profile. Playwright Firefox 157/FFmpeg test software was downloaded separately. **No model weights were downloaded in this correction**, no training/cloud deployment occurred, and ASR/LLM/TTS model identities and synthetic conditioning were preserved. Historical P02/P04 download deviations remain recorded above.

`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 runtimes/session-api/.venv/bin/python scripts/qualify_p04.py` returned 0 and exported [new recovery evidence](../../v2/benchmarks/reports/p04-firefox-recovery.json), binding sources, locks, browser/driver results and the actual received synthetic answer. The previous report remains historical; the exporter now refuses to overwrite an existing report. The new controlled graph proxy is **4,921.3 ms, n=1**, with the same 20 ms observation polling/FFT and uncalibrated device/graph offsets. It neither supersedes the earlier observation as a population statistic nor establishes meaningful acoustic/aligned A/V quantiles. The measured latency shortfall remains open.

**P04 stays `in_progress`; P05 has not started.** The reported Firefox connection failure and independent recovery work are fixed and validated. Completing the phase still requires P03's cached actual Claire runtime/model and an authorized compatible avatar, live speaking video, and the real-user/presentation/pause/echo/cutoff/interruption evidence. The user was asked for those local paths and existing consent/basis/topology/camera records while independent checks continued; none has yet been supplied. **Next executable task:** verify those immutable cached inputs without downloading weights, integrate the real motion/render path, then measure the intended user's meaningful acoustic endpoint→aligned browser presentation and cancellation behavior. Audio-only/static fixtures do not close that gate. Git identity/destination remain the corrected `22elix3r` account and repository above.
