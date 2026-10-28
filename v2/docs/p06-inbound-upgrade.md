# Existing-preview native input upgrade

Local observation, 2026-10-28. The ordinary audio/static preview at
`http://127.0.0.1:18700` and the opt-in Claire vendor preview at
`http://127.0.0.1:18730` now select the qualified
[bounded native microphone runtime](p06-inbound-audio.md). Both report
`inbound_audio_queue: native_fail_closed`. This is an idle, single-host
configuration upgrade with brief downtime, not active-session draining, a
general migration CLI or a hosted release qualification.

The selected descriptor revision is
`ec5d27245548829391ab745b136b4a7899d020cc251a2c771802edbbe6afe233`;
the loaded FFI is
`16eb6e043651eb1c509bf515f512abac2c0e079035cc24928f72d3daa9866d03`.
No installed SDK package or native media worker was replaced. Qwen, ASR, LLM,
Claire and vendor scene identities retain their existing qualification and
limitations.

## Upgrade and rollback observation

Before each switch the private harness checked that no durable session was
PREPARING, READY, ACTIVE or DRAINING. The default preview initially held
76 CLOSED and 3 FAILED sessions; Claire held 5 CLOSED sessions. It collected
repeatable-read fingerprints of every public database table without exporting
rows, then retained exact private configuration bytes and hashes. Existing
credentials, principals, consent records, asset revisions and revocation state
were included in that comparison.

The direct default API was located by its exact command, owner and kernel start
time, then stopped through a revalidated pidfd. Claire used the existing
`tha local down --root v2 --name p06-claire-v2` ownership checks; its database
volume and assets were retained. An actual exit was observed before changing
configuration. This does not establish a POSIX exit code of zero for the API.

While stopped, only `inbound_audio_runtime` was added to each runtime config.
Claire's profile marker was updated to bind that new runtime hash; its other
integrity hashes stayed unchanged. Configuration writes used private temporary
files, fsync and replacement. These two metadata files are not a single atomic
transaction: the harness retains the originals and records a failed startup
for inspection rather than guessing at an automatic rollback.

Each preview completed **upgrade → exact original configuration rollback →
reapply**. Every step reached real provider readiness. Rollback reported
`python_checked_native_unqualified`, and upgrade/reapply reported
`native_fail_closed`. All database table fingerprints remained identical
across each idle restart. The default database retained 79 sessions, 127
principals and its existing lifecycle records; Claire retained 5 sessions and
5 principals. Subsequent browser tests deliberately created additional test
records, so those later database totals are not claimed identical.

The actual private operator harness command, from the repository root, was:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  v2/runtimes/session-api/.venv/bin/python \
  v2/var/p06-claire/migrate_inbound_previews.py upgrade --profile p04
# Same command with rollback, then reapply; repeat for --profile claire.
```

That host-specific harness is retained under ignored `v2/var/`, alongside
private before/after receipts and backups at each profile's
`inbound-upgrade-20261028/`. It is not a supported checked-in operator command.
The [sanitized report](../benchmarks/reports/p06-inbound-upgrade.json) records
receipt/source hashes, configuration-key differences, readiness, database
counts and browser results without credentials, identifiers or raw media.

## Browser boundary

The upgraded default preview passed **8 Chromium checks** and **1 Firefox
check**, using the existing installed project Playwright executable:

```sh
cd v2/web
./node_modules/.bin/playwright test -c playwright.config.ts --reporter=json
./node_modules/.bin/playwright test -c playwright.firefox.config.ts --reporter=json
```

The Chromium cohort includes typed and synthetic-microphone real-provider
playback, interruption, connection/device/permission/autoplay recovery and
cancel during preparation. Firefox connects with loopback ICE disabled,
decodes real Qwen speech and interrupts. This is not a natural human ASR,
voice-quality, physical playback or latency-SLO pass.

Claire's upgraded profile passed **2 Chromium checks in 37.77 s** and
**1 Firefox check in 19.36 s**, using `playwright.claire.config.ts` and
`playwright.claire.firefox.config.ts` with private cohorts
`chromium-native-input-upgrade` and `firefox-native-input-upgrade`.
The real cached Qwen → Claire → CUDA render/NVENC → LiveKit path produced
decoded speech and a final-epoch video callback; typed interruption used new
audio/video tracks, and Chromium also exercised synthetic microphone ASR.
Sender counts and bounds remain distinct from receiver delivery: these browser
checks do not close the existing intermediate-frame gaps.

A strengthened Firefox input test also passed after a controlled **25-second
idle** following synthetic-microphone VAD barge-in, mute and cancellation
acknowledgment. It snapshots pre-send bindings and requires decoded audio matching both the new
text control's returned epoch and the newly appended binding's audio SID,
rejects receiver recovery during the pause, and crosses multiple SSE and
browser-lease heartbeat intervals. Its separate bounded-input reference runs
at port 18760. The one-case run passed in **32.27 s**:

```sh
THA_LOCAL_PROFILE=p06-input-bounded \
THA_INBOUND_COHORT=firefox-idle25-control-qualified THA_INBOUND_IDLE_MS=25000 \
  ./node_modules/.bin/playwright test -c playwright.inbound.firefox.config.ts \
  --grep 'Firefox synthetic microphone' --reporter=json
```

The earlier unmuted continuous-fake-input control-connection failure remains
unexplained. A passing muted-idle fixture does not resolve that different
condition or qualify long natural conversations.

No model/software download, training, global driver/settings change, V1
mutation or cloud deployment was performed for this upgrade checkpoint.
Physical A/V, complete native receiver delivery, independent quality,
Gaussian retarget, human enrollment, resource/capacity and release gates
remain open.
