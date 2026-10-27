# P06 offline local profiles

`tha local init|doctor|provision|up|down` manages an isolated Linux loopback profile using existing verified caches. It creates independent PostgreSQL/SFU containers, database volume, dev identities, API environment, asset storage and process ownership. It shares operator-selected ASR/Qwen/web inputs. The existing P04 preview and its data are preserved.

This qualifies a fresh profile and software environment on the observed Fedora host. It is **not** a fresh bare-machine installation, a containerized API image, native-product integration or a research release. See [native qualification](p06.md), [profile evidence](../benchmarks/reports/p06-local-profile.json) and the implementation status for remaining gates.

From the repository root, with the existing P04 caches, Ollama and pinned PG/SFU images available:

```sh
v2/.venv/bin/tha local init --root v2 --name my-local \
  --from-config v2/var/p04/runtime.json
v2/.venv/bin/tha local doctor --root v2 --name my-local
v2/.venv/bin/tha local provision --root v2 --name my-local
v2/.venv/bin/tha local up --root v2 --name my-local
# Open http://127.0.0.1:18710/ on this computer.
v2/.venv/bin/tha local down --root v2 --name my-local
```

The initial doctor reports an incomplete profile until its separate environment is provisioned. `provision` runs `uv sync --locked --offline` with a per-profile `UV_PROJECT_ENVIRONMENT`; it installed **62 packages from the existing cache** in the exercised profile. Missing software/cache/image inputs fail explicitly. There is no online fallback, model pull, compiler/driver install or default-device change. The offline environment reuses the previously cached unused Faster Whisper Silero ONNX dependency file; this is not a new network download. Actual serving continues to use cached Qwen, CPU ASR/WebRTC VAD and the explicit CPU Ollama model.

Default ports are API 18710, PostgreSQL 15442, SFU signaling 17890, RTC TCP 17891 and UDP 17892. All five can be selected at `init` with `--api-port`, `--postgres-port`, `--signal-port`, `--tcp-port`, `--udp-port`; they must be distinct unprivileged ports. Profiles do not yet share a GPU admission pool. Qualify one active conversation across these development profiles until P07 establishes device-wide capacity.

Configuration/secrets/logs live in `v2/var/local/<name>/`, with a 0700 directory and 0600 metadata. Profile names cannot escape that directory. Existing profiles refuse reinitialization; config digests/ownership are checked before service mutations. Symlinks, public metadata, overlapping operations and changed config are rejected. `doctor` checks ASR hashes, cache/build presence, pinned local image availability and the exact cached LLM digest without loading GPU models. Full Qwen worker/model/voice descriptor hashes are verified by API startup, before admission.

`up` runs digest-pinned Compose with `--pull never`, applies the frozen migrations, starts the API on the configured 127.0.0.1 port and waits for real provider readiness. Native Linux Docker bridge ICE remains the qualified Firefox transport; there is no claim for remote browsers or Docker Desktop. The API runtime is a locked host environment; building a qualified API image remains P06 work.

The API acquires a lifetime lock and records its own PID/start time before provider loading. The launcher holds a separate operation lock; a launcher failure cannot leave an unrecorded API. `down` validates the exact runtime command and kernel start time, opens a Linux pidfd, revalidates ownership and signals that pinned process. It waits for actual exit before stopping its own Compose project. A 20-second drain timeout retains the API record and services for recovery; it does not kill only the API and potentially strand a GPU child. Wrong/reused PIDs are never signaled. The database volume and private assets are always retained. There is no destructive purge option.

The observed qualification used profile `p06-check`, preserving the normal preview at 18700:

```sh
v2/.venv/bin/tha local init --root v2 --name p06-check --from-config v2/var/p04/runtime.json
v2/.venv/bin/tha local provision --root v2 --name p06-check
v2/.venv/bin/tha local up --root v2 --name p06-check
cd v2/web
THA_LOCAL_PROFILE=p06-check npx playwright test --config playwright.local.config.ts \
  --reporter=json > ../var/p06/playwright-local-owned-entrypoint.json
```

**2 browser checks passed in 26.12 seconds**, zero retries: actual typed LLM/Qwen playback with interruption/replacement tracks, and synthetic browser microphone → actual CPU ASR → LLM/Qwen → decoded receiver audio. Tests exercised the real isolated database, SFU, API and cached providers. They do not qualify human microphone quality, voice quality, physical A/V skew or conversational latency SLOs. Only synthetic input was recorded privately.

Both before/after the final restart, the database held migration `p05_0003`, **four CLOSED sessions and four dev principals**. The launcher was deliberately terminated immediately after the API self-recorded, before readiness; the same recorded API survived and became ready. Subsequent `up` recognized it; `down` retired it and its containers while retaining the volume. The original preview remained ready. The qualification profile is currently stopped.

CPU tests cover isolation, exact ownership, foreign/reused PID rejection, actual pidfd shutdown, retained dependencies on drain timeout, configuration tampering, symlinks, concurrent mutation and offline provisioning. The readiness harness now follows each test configuration's explicit loopback origin rather than silently polling port 18700.

An initial restart preflight incorrectly treated TCP TIME_WAIT as an active listener; matching the server's `SO_REUSEADDR` semantics fixed it. A process can clear `/proc` argv during exit before becoming a zombie; shutdown now waits on pidfd readiness instead of re-reading argv in that window. Earlier failed commands remain private evidence. No failure was converted into phase completion.

To export fresh sanitized evidence after actually exercising the same qualification sequence, use `scripts/qualify_local_profile.py`; it checks passing browser results, equal database snapshots, launcher-fault evidence, stopped API and retained Docker volume. Raw config, credentials, transcripts, recordings and identity IDs are excluded from the committed report. P06 still needs native session integration, API image packaging, live virtual output, target conference-client qualification and co-load/resource acceptance; earlier actual-avatar and quality gates remain open.
