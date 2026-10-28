# P06 in-turn encoder rate reduction and recovery

Local observation, 2026-10-29. Three real cached-Qwen/Claire/CUDA/NVENC/SFU
turns retained **324/324** source frames while an isolated encoder ceiling
changed **1.2 Mbps → 300 kbps → 1.2 Mbps**. Source payload rates fell and
recovered in the same turn. This follows the [fixed-ceiling study](p06-publisher-lifecycle-rate.md).
The SDK publish ceiling remained 1.2 Mbps; no network shaping or congestion
controller qualification is implied. Serving configuration remains unchanged.

## Injection and observation boundaries

`--probe-rate-schedule down-up` requires the explicit stage-trace runtime,
live/source tracing and `--probe-bitrate-bps 1200000`. The child receives the
exact additional `THA_CLAIRE_RATE_PROBE_SCHEDULE=down-up` flag. Missing guard,
unknown schedule or a different initial rate is rejected. Serving removes
the inherited schedule flag along with all other diagnostic overrides.

The audio source-frame clock selects the low ceiling at sample **144,000**
(three seconds) and restores the initial ceiling at **288,000** (six seconds).
The GPU owner applies the selected rate before encoding that source frame,
even when no new SDK feedback arrives. It retains the last requested SDK
target and applies the lower of that target and the current experimental
ceiling, within the existing encoder bounds. Actual feedback remains distinct
from an injected ceiling transition. Default serving behavior has no schedule.

V2 probe metadata records the initial accepted setting, each SDK feedback
application and each ceiling transition, with stream samples, source-frame
samples, requested/applied rates, FPS and reason. The two sample counters have
different origins because stream time includes preroll. At most **256** rows
are retained. The parser requires the initial row and both ordered transitions,
consistent clamps, finite/bounded rates and intact row counts. These are
accepted API settings, not hardware-execution acknowledgments. Independent
capture payload sizes measure the resulting encoded output.

## Results

All three turns applied the low ceiling at source sample 144,000 and restored
it at 288,000. Each SDK target at those transitions was **1.2 Mbps**; applied
rates were **300 kbps** and **1.2 Mbps** respectively. This separates an injected
encoder cap from a BWE target reduction. Every turn exited 0 without panic,
with no missing source frame at upload, encoder output, packetizer or receiver.

| Turn | Source seconds 0–3 | Source seconds 3–6 | Source seconds 6–EOF |
|---|---:|---:|---:|
| 1 | 1.2519 Mbps | 353.70 kbps | 1.2417 Mbps |
| 2 | 1.1853 Mbps | 353.70 kbps | 1.2060 Mbps |
| 3 | 1.2315 Mbps | 357.01 kbps | 1.2602 Mbps |

These phase averages include transition transients. Low-phase overshoot is
retained; the result is not strict rate-cap compliance. Full one-second
payload windows and the final partial window remain in the evidence. RTP
overhead, pacer queue delay, constrained-network response, presentation,
natural interaction and perceptual quality remain unqualified.

## Reproduction

The existing opt-in SDK library was reused without rebuilding it or fetching
software/models. The warnings-as-errors worker/native CPU guard build and
descriptor recipe are the same as the [preceding study](p06-publisher-lifecycle-rate.md),
using new output `v2/var/p06-claire/publisher-in-turn-20261029/runtime.json`.
Its revision is `8b3fd625cdad2202c02241954b745a62e394c35692724805c31c104468968b44`.
Actual matrix command, from the repository root:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/runtimes/session-api/.venv/bin/python \
  v2/scripts/qualify_claire_rate_matrix.py --schedule down-up \
  --publisher-runtime v2/var/p06-claire/publisher-in-turn-20261029/runtime.json \
  --receiver-runtime v2/var/p06-claire/inbound-runtime-reason-v1.json \
  --config v2/var/p04/test-runtime.json \
  --engine v2/var/p03/claire-engine-fp32-n1-final \
  --scene v2/var/p03/claire-render-scene \
  --directory v2/var/p06-claire/publisher-in-turn-matrix-20261029 \
  --report v2/var/p06-claire/publisher-in-turn-matrix-20261029.json
```

The runner preserves prior outputs, owns descendant groups through actual
termination, verifies exact runtime/source identities and exports only bounded
diagnostic metadata. Private receiver/user-data logs, credentials and media are
excluded from public evidence. No additional model/software download, training,
global dependency/driver/settings change, V1 mutation or cloud deployment occurred.

The same final worker also passed the **19-case lifecycle cohort** and a
**live-Qwen turn** with every probe and trace disabled, receiving **84/84** and
**324/324** source frames respectively. Both parents exited 0/no panic;
malformed lifecycle inputs retained their expected failure exits. Admission was
reacquired and publication identities stayed fresh. These are Python-receiver
checks, not browser or physical-presentation acceptance.

Actual final offline commands, from `v2/`:
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q --junitxml=var/p06-claire/publisher-in-turn-core-tests.xml`
— **394 passed in 7.69 s**. The equivalent API command, from
`v2/runtimes/session-api/`, with
`--junitxml=../../var/p06-claire/publisher-in-turn-api-tests.xml` passed
**46 tests in 4.15 s**. The native CPU guard passed. Review found no remaining
critical/important issue and independently passed 43 CPU/parser/process checks
plus the serving-isolation check. All three previews reported ready afterward.

[Sanitized evidence](../benchmarks/reports/p06-publisher-rate-transition.json)
binds the three injected-control turns, both probes-disabled cohorts, source/
runtime/model identities and validation receipts. Recorded preflight waveform
hashes remain labeled as preflight; live synthesis has its own runtime binding.

Next: bind the qualified SDK policy and lifecycle/live evidence into an explicit
serving descriptor, then exercise actor/browser cancellation and recovery before
an idle research-preview rollout.
P02–P06 stay in progress; this test closes no physical A/V, human quality,
capacity, constrained-network or release gate.

The [serving-policy follow-up](p06-publisher-serving.md) now binds fresh owner
receipts to exact probes-disabled reports and deploys this candidate to the
Claire preview after browser and idle rollback checks.
