# P06 qualified publisher policy in the Claire preview

Local functional observation, 2026-10-29. The Claire research preview at
**http://127.0.0.1:18730/** now uses the qualified opt-in external-rate-controller
SDK policy. The normal audio/static preview at 18700 remains separate. This
deploys the correction investigated in the [stage trace](p06-publisher-stage.md),
[fixed-ceiling](p06-publisher-lifecycle-rate.md) and
[in-turn encoder-control](p06-publisher-rate-transition.md) studies.

The isolated profile and existing preview each passed two Chromium journeys
and one Firefox journey. Typed interruption resumes with new tracks; Chromium
also exercises synthetic microphone → CPU ASR/LLM → Qwen → Claire → browser.
All ten observed native workers loaded the selected libraries, enabled the
validated trust flag and omitted diagnostics. Browser checks observe decoded
speech and final-epoch video callbacks, not physical A/V or complete
intermediate-frame presentation.

## Explicit serving selection

Existing descriptors default to `sdk_default`. The new
`publisher_rate_control: external_trusted_local_research` mode requires:

- The tested SDK build receipt matching the pinned Rust revision, Cargo lock,
  opt-in patch and actual FFI hash.
- The same worker, C++/FFI libraries, bridge, PTX, engine, scene and buffer policy
  across probes-disabled live and full 19-case lifecycle reports.
- Complete recorded/live source-frame receiver observations, expected fault
  failures, owned backpressure cancellation and reacquired admission.
- Clean qualifier-owner exits bound to each exact report's SHA-256 and valid
  qualifier identity. Receipts cannot be mixed across reports.
- The same publisher inventory in the completed three-turn injected encoder
  reduction/recovery study, retaining its network limitations.

The snapshot includes six evidence files and the normal model/runtime/host
dependency inventory: **189 files**. Loading validates their hashes. The actor
clears inherited diagnostic/trust flags, then sets exactly
`THA_LK_TRUST_EXTERNAL_RATE_CONTROL=1` only for the validated policy. It enables
no capture, stage or rate probe. Asset-revision and vendor-identity restrictions
remain in force.

Review exposed that the initial clean-exit receipts did not identify report
bytes. That form is rejected. A new owned runner observed actual termination,
hashed the report/log and bound qualifier identity. Fresh **19-case lifecycle**
and **live-Qwen** runs completed with exit 0/no panic, receiving **84/84** recorded
and **324/324** live source frames. Regression tests reject mixed receipts and
missing source identities. Subsequent work stayed in the main thread as requested.

Selected descriptor revision:
`3b560d0c635991923560486636cce1339678742084b6d7523995a92723e925ae`.
Worker SHA:
`7c9ee974627a5b465125c4dff808abd09db0ac22b3c1a2718fee1f2dcb8588d9`.
FFI SHA:
`084e4f30b6748b3c2434eafedbef142856b56bbb8cc0d97c5353992dcb1e1188`.
The API's separate fail-closed microphone FFI remains
`16eb6e043651eb1c509bf515f512abac2c0e079035cc24928f72d3daa9866d03`.

## Actual qualification and pinning

From the repository root, with existing artifacts and new output paths:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/runtimes/session-api/.venv/bin/python \
  v2/scripts/qualify_claire_publisher_lifecycle.py \
  --publisher-runtime v2/var/p06-claire/publisher-in-turn-20261029/runtime.json \
  --receiver-runtime v2/var/p06-claire/inbound-runtime-reason-v1.json \
  --config v2/var/p04/test-runtime.json \
  --engine v2/var/p03/claire-engine-fp32-n1-final \
  --scene v2/var/p03/claire-render-scene \
  --directory v2/var/p06-claire/publisher-serving-bound-20261029 \
  --report v2/var/p06-claire/publisher-serving-bound-20261029.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/.venv/bin/python \
  v2/scripts/describe_claire_media.py \
  --engine v2/var/p03/claire-engine-fp32-n1-final \
  --scene v2/var/p03/claire-render-scene \
  --qualified-live-report v2/var/p06-claire/publisher-serving-bound-20261029/live.json \
  --publisher-runtime v2/var/p06-claire/publisher-in-turn-20261029/runtime.json \
  --trust-encoded-rate-control \
  --lifecycle-report v2/var/p06-claire/publisher-serving-bound-20261029/lifecycle.json \
  --rate-report v2/var/p06-claire/publisher-in-turn-matrix-20261029.json \
  --live-owner v2/var/p06-claire/publisher-serving-bound-20261029/live-owner.json \
  --lifecycle-owner v2/var/p06-claire/publisher-serving-bound-20261029/lifecycle-owner.json \
  --output v2/var/p06-claire/serving-publisher-trusted-bound-20261029/runtime.json
```

The runner preserves prior evidence, owns descendant groups through termination
and records post-report failures. Pinning copies qualified binaries/evidence
and checks dependencies; it never fetches/builds a model. Snapshot names include
the FFI hash, preventing silent reuse between SDK variants.

## Isolated profile, rollout and rollback

Independent `p06-publisher` at **18790** uses new database/SFU credentials and
ports 15470/17970/17971/17972, explicit cached inputs and the verified shared API
environment. Offline provision/doctor and startup passed. This is a fresh local
profile on the qualified host, not a bare-machine image installation.

Commands from `v2/web/`, using the existing project executable:

```sh
THA_CLAIRE_PROFILE=p06-publisher THA_CLAIRE_COHORT=chromium-trusted-isolated \
  ./node_modules/.bin/playwright test -c playwright.claire.config.ts --reporter=json
THA_CLAIRE_PROFILE=p06-publisher THA_CLAIRE_COHORT=firefox-trusted-isolated \
  ./node_modules/.bin/playwright test -c playwright.claire.firefox.config.ts --reporter=json
```

They passed **2 in 44.33 s** and **1 in 21.92 s**. A private host-specific wrapper
owned the browser process group and sampled children of the exact API owner,
checking `/proc` library mappings and selected flags. All **3 Chromium / 2
Firefox** workers matched. Raw environments and publication IDs were not exported.

The existing preview completed **upgrade → exact original-config rollback →
reapply**, with idle-session checks. Only `native_media_runtime` changed;
its integrity marker was updated. All **15** database-table fingerprints stayed
identical across the three restarts, including the original **8** sessions.
Rollback selected original descriptor
`e721307c56c2e4d69cd453b403ccbf2f049802fdebf4bc0b42df58f1b5cd5559` and disabled
trusted control; upgrade/reapply selected the qualified descriptor. Every step
reached readiness and retained bounded microphone input.

Actual private harness: `v2/runtimes/session-api/.venv/bin/python
v2/var/p06-claire/migrate_publisher_preview.py upgrade`, then `rollback` and
`reapply`. Exact-byte backups and receipts remain under the profile's
`publisher-upgrade-20261029/`. This idle switch has brief downtime; the two
configuration files are not one crash-atomic transaction, and atomic admission
drain/general hosted migration are not qualified.

On the upgraded preview, the same commands with
`THA_CLAIRE_PROFILE=p06-claire-v2`, cohorts `chromium-trusted-preview` and
`firefox-trusted-preview`, passed **2 in 39.80 s** and **1 in 20.20 s**. Its
additional **3 / 2** workers matched the runtime/flags. Browser tests deliberately
add database records after the idle comparisons.

## Validation and remaining gates

From `v2/`,
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q --junitxml=var/p06-claire/publisher-serving-core-tests-final.xml`
passed **417 in 8.04 s**. From `v2/runtimes/session-api/`, the equivalent API
command with `--junitxml=../../var/p06-claire/publisher-serving-api-tests-final.xml`
passed **47 in 4.65 s**. From `v2/web/`,
`./node_modules/.bin/tsc --noEmit && npm test -- --run` passed type checking and
**2** unit tests. No new native/Rust build was needed.

[Sanitized evidence](../benchmarks/reports/p06-publisher-serving.json) binds
source/runtime, bound-owner, browser and rollback receipts without credentials,
media or publication IDs. Ports 18700/18730/18760/18790 reported ready. No
model/software download, training, global setting/driver change, V1 mutation
or cloud deployment occurred. Existing vendor terms/research scope remain.

The observed SDK frame-dropper problem is corrected in this local profile.
Physical alignment/suppression, natural unmuted conversation, independent
quality, Gaussian retarget, human enrollment, sustained resource/capacity,
images/live virtual devices, constrained networks and release gates remain
open. P02–P06 stay in progress. Next: sustained joint-resource/cancellation
qualification and the remaining P06 packaging/virtual-device work.
