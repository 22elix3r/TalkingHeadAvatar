# P06 device admission and worker lifetime

One admitted conversation/qualification worker owns the managed device-0 lease for the selected native-runtime directory. `tha local` profiles share the original absolute Qwen runtime descriptor, so they share this namespace even with separate API processes and databases. This protects the existing session before another profile or standalone provider can load a competing Qwen worker. It is a coarse N=1 guard, not shared-weight pooling or measured multi-session capacity.

`load_native_config` derives `gpu-device-0.lock` beside the operator-owned runtime descriptor. An actor acquires it before provider warmup and retains the parent reference until source cleanup completes. A standalone provider acquires the same lease when opening its worker and releases it on close. Cancel/EOF keep the lease while model weights remain resident. GPU children inherit the same open-file description via `pass_fds`; losing/releasing the parent reference therefore does not admit another owner while an inherited child still holds it. Release closes the reference without `LOCK_UN`; the lock inode is never unlinked/replaced.

Kernel flock ownership is authoritative. Persisted metadata is diagnostic only, hashes the session identifier and contains no voice/text/PCM. Symlinks, hardlinks, public/foreign files, writable shared directories and pre-existing unrelated/corrupt private data are rejected without truncation. A filename/PID is not proof that the lease is currently held. Separate descriptor directories, multiple GPU UUIDs, other user accounts and unmanaged CUDA programs require a qualified device broker/profile; this guard does not isolate them.

Provider startup is serialized. Cancellation during asynchronous process creation retains the actual child handle and waits for its retirement rather than losing an unrecorded worker. Session close releases its parent lease even when durable-close persistence fails; a still-living inherited GPU child continues to hold admission closed. The browser receives an actionable `gpu_capacity_busy` error when another managed task is active. The second profile's asynchronous session attempt terminates before model process creation; HTTP 202 reservation is not successful GPU admission.

The existing preview was idle and briefly restarted to load the guard; no active conversation was interrupted. Config/model/voice/checkpoint revisions and source assets were preserved. No dependency or model download, training, global settings change or V1 change occurred.

## Actual qualification

- **241 core checks passed in 5.14 s**, including real kernel lock inheritance after parent release, retained inode identity, private-file preservation, provider cancellation/EOF residency, externally owned session lease retention, duplicate startup rejection and cancellation racing child creation. These CPU protocol fixtures never initialize models/CUDA.
- **20 runtime/PostgreSQL checks passed in 4.00 s**, including busy-device failure before voice loading, retained competing ownership and idempotent close despite durable-store failure.
- **One real two-profile browser scenario passed in 22.74 s**: primary session loads actual cached Qwen; a standalone competitor and second profile are rejected; primary still produces decoded speech; after primary close the second profile acquires admission and produces decoded speech. The real Qwen child was observed holding the inherited file descriptor. The competitor's spawn function is instrumented to fail if reached, so a broken guard cannot accidentally load a second model during this test.
- Existing real-provider regressions: **8 Chromium checks in 56.70 s**, **one Firefox 157 check in 8.20 s**, zero retries. Web unit checks (two), TypeScript and production build passed. The existing deferred client-chunk size warning remains.

With the normal preview ready and existing offline caches, reproduce the isolated second-profile check from the repository root:

```sh
v2/.venv/bin/tha local up --root v2 --name p06-check
cd v2/web
npx playwright test --config playwright.device.config.ts --reporter=json \
  > ../var/p06/playwright-device-admission-final.json
```

`check_device_admission.py --runtime ... --output ...` is a no-spawn competitor probe and requires an already admitted real session. It validates existing model/worker/voice hashes, observes the real inherited native-worker descriptor, then checks rejection before the process-creation boundary. It is not a speech benchmark or an independent model evaluation.

The [sanitized report](../benchmarks/reports/p06-device-admission.json) records the CPU/runtime JUnit evidence, browser results, source fingerprints and scope. The exporter `scripts/qualify_device_lease.py` refuses failed checks, missing real-worker inheritance/handoff evidence or a changed preview source snapshot. It also checks that the lease can be reacquired after the tested sessions close. Private session/holder IDs, logs and raw observations are excluded.

Native media consumers will inherit the same actor-owned lease. Actual native Qwen/render/encode co-load, physical A/V quality, memory/admission limits beyond N=1, worker-loss behavior across the whole product and the capacity sweep/eight-hour soak remain unqualified. This checkpoint does not close P06/P07 or the existing quality gates. The temporary second API profile is stopped; the normal preview remains available.
