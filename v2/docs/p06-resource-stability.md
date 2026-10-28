# P06 joint-resource and cancellation stability

Smoke checkpoint, 2026-10-29. The new explicit browser soak keeps one typed
session actor, uses real cached CPU LLM/Qwen/Claire/native playback and alternates
complete replies with interruption after decoded speech. The final short cohort
passed **7 turns: 5 complete and 2 interrupted**, then confirmed durable close
and no remaining API children. A separate 1,800-second cohort was started;
its gate is not satisfied by this smoke. Consult the current status/receipt
before claiming the longer run finished.

The workload is typed and closed-loop. It does not measure human endpointing,
natural microphone conversation, physical A/V, open-loop admission capacity,
independent quality, latency SLOs or an eight-hour soak. The selected profile
is the already qualified [publisher runtime](p06-publisher-serving.md) at
18790; the preview configuration is unchanged. The single-GPU reservation
protects the active test from competing conversations.

## Instrumented boundaries

Each text control's exact response epoch must receive fresh audio/video SIDs,
decoded audio for that SID/epoch, and the corresponding video callback. Every
fifth turn interrupts after decoded audio. The test records whether sender EOF
was already seen, requires a successful interrupt response and local audio
detachment/mute, then requires new publications on the next answer. Both smoke
interruptions preceded sender EOF; this is not an acoustic suppression-time claim.

Completed turns retain sample/padding conservation, source-frame accounting and
native queue maxima. The observed maxima were **48,000 audio samples, 13 motion
packets and 24 poses**, with zero reported render skips in this short cohort.
Cancelled turns do not have the same complete EOF statistics. Events are bounded
to 256 per attempt and the fixture has a 1,024-turn guard. Per-turn JSONL is
written before proceeding, and failed attempts remain accounted for.

The read-only observer samples device memory, GPU/encoder utilization, temperature,
clocks/power, host memory and exact-owner API/Qwen/media RSS, FDs, threads and CPU
ticks. Native mappings/flags are checked when initialized. Queries are sequential
with a target 200 ms pause after each cycle; these are sampled maxima, not
guaranteed peaks or simultaneous per-stage measurements.

The final smoke collected **302 samples**, **266** with positive Qwen+media GPU
attribution, and zero query failures. Three starting/retiring process observations
had no GPU attribution; they remain unknown and do not count as co-residency.
One Qwen process survived all seven media-worker lifetimes; all seven media
policies were observed and all children retired at close.

Sampled maxima: **9,687 MiB total device use**, **4,376 MiB Qwen** and **1,203 MiB
media**. The media process combines Claire, rendering, conversion, NVENC and SDK
resources; it is not an isolated encoder measurement and does not pass the
512 MiB encoder budget. Per-role maxima must not be added as a simultaneous peak.
Desktop/other use remains in the device total. The developer desktop and CPU
tooling/packaging context is explicit; this is not an isolated capacity baseline.

## Actual commands and evidence

From the repository root, with the existing profile idle, the final smoke used:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/runtimes/session-api/.venv/bin/python \
  v2/scripts/qualify_claire_session_soak.py --profile p06-publisher \
  --cohort chromium-resource-smoke-bound-20261029 --seconds 60 \
  --context desktop_and_cpu_packaging \
  --directory v2/var/p06-claire/resource-smoke-bound-20261029 \
  --report v2/var/p06-claire/resource-smoke-bound-20261029.json
```

The 30-minute run uses the same command with `--seconds 1800`, cohort
`chromium-resource-30m-20261029` and corresponding new directory/report paths.
Startup and cleanup are retained separately from per-turn events. The last turn
may finish after the requested interval; the final smoke observed 69.26 seconds
from workload start through confirmed close. It never pools that interval into
an annotated speech-endpoint latency distribution.

An initial instrumented browser smoke completed, but the exporter failed when
it compared missing GPU attribution with an integer. Its raw observations remain
private. A CPU regression now treats unknown attribution as unavailable;
the corrected final smoke above completed successfully. Earlier shorter fixture
and failed-export versions remain separate from the final source fingerprints.

Final core validation: from `v2/`,
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q --junitxml=var/p06-claire/resource-probe-core-tests.xml`
— **431 passed in 8.26 s**. From `v2/web/`,
`./node_modules/.bin/tsc --noEmit && npm test -- --run` passed type checking and
**2** unit tests. No serving API behavior changed, and no new API-suite result
is inferred from this harness-only change.

[Sanitized smoke evidence](../benchmarks/reports/p06-resource-smoke.json) binds
the exact source/runtime/receipt identities and resource/queue observations.
Raw telemetry, publication IDs and browser events stay private. No model/software
download, training, global change, V1 mutation or cloud deployment occurred.
P02–P06 remain in progress; the full sustained result and remaining packaging,
virtual-device, physical presentation and quality gates stay open.

## 1,800-second cohort outcome

The separate sustained run **failed**, with 68 complete and 18 interrupted
turns before attempt 87 stopped on the retained native code `Claire motion
underrun; retire epoch`. Its measured workload-to-close interval was 931.58 s,
short of 1,800 s. Source/runtime pins stayed stable; close was confirmed and
no API children remained. The observer recorded 3,757 samples, zero query
failures and a sampled device maximum of 9,753 MiB. Completed-turn queue maxima
stayed at 48,000 audio samples / 13 motion packets / 24 poses, with zero reported
render skips. These prior successful turns do not turn the failed run into a
pass. CPU-only image staging/build overlapped; it is context, not an established
cause. The guard identifies a motion-readiness failure, but its underlying delay
remains unresolved. Numeric readiness/owner timing is needed before changing
buffer policy or rerunning the full cohort.

[Failure evidence](../benchmarks/reports/p06-resource-30m-failure.json) retains
all attempt accounting, runtime/source/receipt hashes and bounded fault/resource
metadata. The raw events, detailed logs and private identifiers remain excluded.
The earlier smoke and its report remain distinct.

After explicit close, a fresh Chromium typed/microphone recovery cohort passed
2 checks in 46.45 s; all three native workers matched the selected SDK policy.
This qualifies post-close reuse in those short journeys, not a fix for the
underrun or a passing sustained cohort. All four local profile endpoints were
ready afterward.

## Instrumented retry

[Bounded readiness diagnostics](p06-readiness-diagnostics.md) now retain numeric
source/motion/owner state on a terminal guard failure. A deliberate input-starvation
control, native lifecycle/live/rate checks and a new seven-turn isolated browser
smoke passed. The new 1,800-second cohort `chromium-readiness-30m-20261029` is
running without the earlier overlapping packaging build. The original failed
cohort remains a failure; diagnostics and smoke do not close sustained stability.
