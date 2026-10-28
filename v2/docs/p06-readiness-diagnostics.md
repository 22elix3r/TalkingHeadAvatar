# P06 motion-readiness fault diagnostics

The first 1,800-second browser cohort failed with a motion underrun. That code
identified the guard that retired the epoch, but the previous cleanup retained
no owner/readiness state. Its cause remains unresolved. See the
[failed sustained cohort](p06-resource-stability.md#1800-second-cohort-outcome).

The worker now emits one numeric `THA_CLAIRE_READINESS_V1` line when either
active-audio or motion readiness fails. Audio/input counters are read under the
existing mutex; GPU-owner counters are independent atomic observations. This is
a best-effort fault snapshot, not a simultaneous GPU trace or a presentation
measurement. The existing guard conditions, 500 ms startup, 800 ms motion lead,
queue bounds and epoch retirement behavior are unchanged.

The snapshot records the consumed/received/queued audio counts, queued motion
packets, offered 16 kHz motion count, available 48 kHz pose horizon, input EOF,
owner stage/age, last/max push/render/capture durations, source/skipped frames,
known owner failure and maximum audio-clock lateness. Owner stages are 0 waiting,
1 push/metrics, 2 render/encode, 3 SDK capture and 4 rate control. Stage age can
include idle time after completion; consult the stage and progress age together.
It does not contain PCM, coefficients, transcripts, paths or identifiers.

Python cleanup accepts only the exact version, field count, unsigned decimal
syntax and per-field bounds. Malformed lines and arbitrary vendor text are
excluded. The numeric record joins the existing fixed fault code in an
operator-private 0600 JSON receipt; the epoch's temporary bootstrap/log directory
is still removed. The browser soak also retains its bounded observation window
on a failed attempt, with boolean alert/audio state and overflow/recovery counts.
It excludes DOM text, transcripts and arbitrary exception messages.

## Deliberate starvation control

The qualifier's explicit `--starve-motion-input` mode sends one second of paired
cached synthetic PCM and withholds further input and EOF. It requires a native
error/exit 1, audio remaining queued and an exhausted motion horizon, plus
successful admission reacquisition after cleanup. This validates instrumentation
and fail-closed behavior; it does not reproduce the original upstream delay.
It never injects a stall into a serving worker.

The first control observed 36,960 consumed audio samples, 48,000 received,
11,040 queued, no queued motion packets and 16,000 offered motion samples. The
pose horizon ended at 36,800, with input EOF false, owner stage waiting and no
known owner failure. The guard retired the epoch and the qualifier returned 0.
Timing values describe this one observation only.

The instrumented binary is pinned separately from the preview's existing runtime.
Its SDK, Claire bridge, PTX, engine and scene remain unchanged. A separate
probes-disabled 19-case lifecycle/live-Qwen qualification and injected encoder
rate checks are required before selecting it in the isolated browser profile.
No model download or source-asset mutation is required.

## Qualified native checkpoint

The new worker SHA-256 is
`006cd0b386b2050b9c8f13d11e59140de506f1aaa1c24c74ab4fcb960717d99e`.
Publisher probe descriptor `cc68cca11d1b3cbdf70f250ad95ae26498f4e484a21ae6a1306a39a020823a1c`
and serving descriptor `6dfb44e952bbe05788b2fcbe0d47c142a30486b8cedf9efc3bbcd9e1486b37bc`
remain private inventories. All 19 lifecycle cases passed; the recorded fixture
received 84/84 source frames and live Qwen received 324/324. Owned exits were 0,
without observed panic; lifecycle/live source and stage/rate probes were disabled.
Three separate injected down/up encoder-ceiling turns each received 324/324.
Those injected ceilings do not qualify constrained-network rate control.

Only the idle `p06-publisher` profile was switched. All 15 durable table
fingerprints matched across the restart, and readiness passed. The user preview
at 18730 keeps descriptor
`3b560d0c635991923560486636cce1339678742084b6d7523995a92723e925ae`.
The image built at the previous checkpoint still contains its earlier source
snapshot; it is not silently relabeled as this instrumented runtime.

From the repository root, after rebuilding the first-party worker and pinning
its isolated publisher inventory, the positive control used:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/runtimes/session-api/.venv/bin/python \
  v2/scripts/qualify_claire_session_worker.py --starve-motion-input \
  --publisher-runtime v2/var/p06-claire/publisher-readiness-20261029/runtime.json \
  --trust-encoded-rate-control \
  --receiver-runtime v2/var/p06-claire/inbound-runtime-reason-v1.json \
  --config v2/var/local/p06-publisher/runtime.json \
  --engine v2/var/p03/claire-engine-fp32-n1-final \
  --scene v2/var/p03/claire-render-scene \
  --directory v2/var/p06-claire/readiness-positive-20261029 \
  --report v2/var/p06-claire/readiness-positive-20261029.json
```

This command was run under the existing process-group owner helper, which
retains the actual exit and report/log hashes. `qualify_claire_publisher_lifecycle.py`
then ran lifecycle/live checks with the same config/engine/scene/runtime paths,
directory `readiness-lifecycle-20261029` and corresponding `.json` report.
`qualify_claire_rate_matrix.py --schedule down-up` used directory
`readiness-rate-20261029` and its separate report.

Core validation from `v2/`: offline `.venv/bin/python -m pytest -q
--junitxml=var/p06-claire/readiness-core-tests.xml` — **438 passed in 7.96 s**.
From `v2/runtimes/session-api/`, the same offline pytest command with
`--junitxml=../../var/p06-claire/readiness-api-tests.xml` — **47 passed in 5.57 s**.
Web `tsc --noEmit` and `npm test -- --run` passed, with **2** unit tests.
An initial core invocation from the repository root selected unrelated runtime
and cached-source tests and failed collection; the correctly scoped V2 run is
the result above. No V1 test pass is inferred.

[Sanitized evidence](../benchmarks/reports/p06-readiness-diagnostics.json) binds
the code, runtime, owner receipts and validation. The original sustained-failure
cause and resource gate remain open. No model/software download, training,
V1 mutation, global setting change or external deployment occurred. All work
stays in the main thread.

The isolated 60-second browser smoke passed **7 attempts: 5 complete and 2
interrupted**, with confirmed close, no API children left and stable source/runtime
pins. It measured 70.96 seconds from workload start through close. All seven media
lifetimes matched the selected policy; 313 resource samples included 269 with
positive Qwen+media GPU attribution and zero query failures. Sampled GPU maxima
were 9,328 MiB total, 4,376 MiB Qwen and 1,215 MiB combined media. These are not
simultaneous per-stage peaks or an isolated encoder allocation.

The separate 1,800-second retry is running with cohort
`chromium-readiness-30m-20261029`, profile `p06-publisher`, context `desktop_only`,
and directory/report `v2/var/p06-claire/readiness-30m-20261029{,.json}`. It uses
the existing soak command with `--seconds 1800`; no packaging build overlaps.
It is not yet a passing sustained result. A clean retry would still not establish
the cause of the original failure.
