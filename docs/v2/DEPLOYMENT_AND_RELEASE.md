# Deployment, operations and release specification

Revision 2026-10-26. These are implementation requirements and acceptance procedures, **not commands or configuration already available in V2**. The [E2E specification](E2E_IMPLEMENTATION_SPEC.md) defines service/API behavior; the [backlog](IMPLEMENTATION_BACKLOG.md) names deliverables. Infrastructure evidence: [research addendum](DEPLOYMENT_RESEARCH.md).

**Current release target is research deployment.** Noncommercial models may be primary providers. Use [independent model quality/Q0](MODEL_QUALITY_BENCHMARKS.md) to select them; record terms and research access now, defer commercial clearance. Engineering qualification (timing, quality, security, recovery) and commercial-distribution permission are separate statuses. An NC label must not automatically reject a research profile.

Implement the selected Qwen profile through [P00–P09](IMPLEMENTATION_PHASES.md); track evidence in [implementation status](IMPLEMENTATION_STATUS.md). Other TTS families are deferred.

## 1. Supported deployment profiles

| Profile | Services and storage | Qualification |
|---|---|---|
| CPU functional CI | API, PostgreSQL, synthetic media/provider fixtures, local blob store; browser integration tier adds LiveKit | Correctness only; no neural quality/latency claim |
| Fedora local | Web/API, PostgreSQL, one session runtime, CPU ASR/turn pools, selected Qwen GPU TTS worker, CPU LLM or explicit remote LLM, one GPU media worker, LiveKit, local blob volumes | RTX 4070 Ti SUPER 16 GB / Ryzen 7600X / 32 GB; measured total memory and CPU thread budgets |
| Single GPU host | Same logical components in separate containers, TLS edge, operator-selected object storage, OIDC, TURN and telemetry | First deployable hosted profile; admitted N determined by measurements |
| Multiple hosts/GPUs | Redundant API, PostgreSQL/object-store durability, distributed LiveKit/Redis, independent speech and media pools, router/admission | Regional affinity, failure isolation, drain and qualified GPU/encoder combinations |
| Larger-GPU research quality | Independently selected speech/video references, isolated from the local profile; optional remote providers explicitly identified | Drama/LiveAvatar feasibility and latency qualified on actual hardware; never advertise their quality and another profile's latency as one result |

Do not claim the reference host can run ASR, cloned TTS, all storage/observability services, enrollment and the full GPU stack simultaneously without measurement. Enrollment is stopped or scheduled on separate hardware during serving. The developer profile may omit dashboards and use a hosted LLM; its manifest must disclose that. Local/offline operation means no network after explicit provisioning, and requires a separately passing all-local profile.

## 2. Packaging deliverables

Implement `v2/deployment/compose.yaml`, profile overlays, `.env.example` containing names only, config schemas and a `tha` CLI. Provide immutable CPU API/session, speech-provider, GPU media and enrollment images, plus static browser assets. Separate build/runtime stages and mount verified model/asset caches read-only during inference. Keep enrollment writable staging separate from serving storage.

A supported release supplies a tested matrix: host driver minimum/actual, GPU architecture, OS/container base, Python, PyTorch/CUDA, A2F/TensorRT, native SDK/FFI, NVENC, gsplat, browser SDK and codecs. Build using pinned sources and a reproducible toolchain. Hash binary dependencies and model artifacts; store their license texts and notices. Do not fetch latest releases at build or container startup. Engines are rebuilt for qualified hardware/runtime changes rather than assumed portable.

Proposed CLI acceptance interface (to implement):

```text
tha doctor --profile local
tha artifacts verify --manifest <file>
tha artifacts provision --manifest <file>
tha config validate --profile <name>
tha db migrate --check
tha up --profile local
tha smoke --profile local --fixture <id>
tha benchmark --scenario <id> --sessions 1,2,4,8,16
tha drain --worker <id>
tha diagnostics export --redacted
tha down
```

`doctor` is read-only: driver/runtime/device/encoder capabilities, ports, cache space, memory and dependency availability. `provision` is an explicit network/terms-aware step, resumable with size/hash verification; it does not accept licenses for the operator implicitly. `up` must fail with an actionable reason if artifacts/terms are absent; it must not train or download gigabytes unexpectedly. `down` drains and preserves volumes by default. Destructive volume deletion is a separate explicit operation.

Fedora must retain SELinux and host settings. Qualify Podman/CDI or Docker/NVIDIA runtime explicitly; document device permissions and narrow volume labels. Never solve GPU permission failures by silently disabling SELinux globally or mounting the entire host. Rootless/virtual-device support is a profile-specific result, not an assumption. Virtual camera/microphone setup belongs to a documented host adapter, not a privileged container startup hook.

## 3. Networking and trust boundaries

Proposed exposed surface for the first hosted profile:

| Surface | Routing | Rule |
|---|---|---|
| App/API and LiveKit signaling | TLS 443, reverse proxy to private HTTP services | Authenticated API, WebSocket support, bounded bodies/timeouts |
| ICE media | Qualified LiveKit UDP mux, initially 7882/UDP, or configured UDP range | Advertise reachable public/NAT address; test both directions |
| ICE TCP fallback | 7881/TCP if selected | Distinct from ordinary HTTPS reverse proxy |
| TURN | Qualified UDP 3478 and TURN/TLS endpoint | TLS 443 on a dedicated address/name or correctly configured L4 routing; do not collide blindly with app HTTPS |
| Database, RPC, metrics, model admin | Private network/socket only | No public listen without explicit policy |
| Enrollment uploads | Short-lived scoped object grants or bounded API stream | Size/type/hash enforcement and tenant/purpose prefix |

Exact ports are configuration, not hardcoded application constants. Verify DNS, certificates, ICE candidate address, firewall and relay capacity from an external network. TURN must use authenticated temporary credentials and must not become an open relay. Forced-relay tests are mandatory; localhost success is insufficient. If WHIP fallback is selected, declare its separate ingress service/ports and confirm transcoding is disabled for the qualified profile.

Use separate staging and production databases, rooms, artifact namespaces and credentials. Test providers use explicit nonproduction configuration. Never run a load generator against production accidentally; require an environment identifier and capacity ceiling in scenario manifests.

## 4. Startup, readiness and admission

Startup sequence: validate configuration/schema and secrets → verify model/asset hashes and terms → establish database connectivity/migration compatibility → initialize shared engines → warm supported shapes/voices → allocate/test encoder surfaces → connect media signaling → execute a deterministic internal smoke frame/audio sequence → register worker capacity → mark selected profiles ready.

`live` means the process/event loop is responsive. `ready` means it can accept new work for declared profiles. A saturated or draining worker is not ready for new sessions but should not be killed by liveness. A model-loading startup probe has a measured generous deadline; repeated short liveness restarts must not create a cold-start storm. Advertise readiness per profile/asset state rather than one global “GPU available” flag.

Use measured memory/encoder/thread/deadline reservations and fencing from the E2E spec. Session TTL, disconnected grace and maximum duration are explicit product configuration. The initial hosted maximum session duration is 30 minutes with visible warning and a deliberate extension policy; this bounds drain and resource retention. No active session is evicted merely to increase benchmark throughput.

Keep model caches and warm pools bounded. New avatar loads reserve transient memory before loading. Offline provisioning/building never runs on the live critical path. Under pressure stop speculation/debug work first, then apply declared video degradation, then reject new sessions. Record degraded mode duration so availability cannot hide permanently poor video.

## 5. Observability and service objectives

The original sub-second response and quality targets remain conditional on the qualified cohort. Add proposed operational launch objectives, to ratify after staging evidence:

| Objective | Initial target / definition |
|---|---|
| Service availability | 99.5% over 30 days for valid session-create requests completing readiness within the declared warm/cold deadline; capacity rejections count against service availability, caller quota/invalid requests reported separately |
| Warm attach | p95 ≤2 s for already cached selected assets, including admission/room join; separate from turn latency |
| Useful first response | Original warm p95 <1 s and p99 ≤1.5 s with cutoff/quality gates; failures remain violations |
| Resource stability | Eight-hour qualified-load soak without monotonic leak, unexpected process restart or stale cross-session media |
| Revocation | Proposed p95 ≤5 s under reachable control service to stop server-controlled use and reject new grants; partitioned workers stop by their conservative 10 s lease expiry, not the 5 s target; downloads and backup purge have separate limits |
| Recovery | Single worker failure visible within 5 s; no stale replay; optional replacement session ready within 30 s if spare warm capacity exists |
| Backup recovery | Initial metadata RPO ≤24 h and RTO ≤4 h from a tested backup; tighten before stronger availability claims |

These are product targets, not measured uptime or guarantees. Define maintenance accounting and operator coverage before advertising an SLA. A single workstation does not provide redundant hosted availability.

Dashboards: first-response distributions by immutable profile/cohort; endpoint false-cutoff rate; TTS first horizon; output skew/underruns; queue age; active/admitted/rejected sessions; GPU/RAM/CPU/encoder occupancy; OOM/restarts; ICE failures/relay ratio; cold startup; enrollment stage failures; consent purge lag; and release versions. Record exemplar traces without biometric payloads. Avoid session IDs as unbounded metric labels; use them in protected traces.

Alert on sustained deadline/error-budget burn, unavailable profiles, approaching admission reserves, lost worker leases, repeated decoder failures, database/object-store failures and overdue revocations. Give every alert an owner, relevant dashboard and remediation. A transient GPU utilization spike alone is not an incident.

## 6. Failure runbooks

| Symptom | Immediate action | Recovery and evidence |
|---|---|---|
| Audio works, face stalls | Preserve audio; show declared degraded state; stop new visual sessions on affected worker | Inspect render/encode queue, GPU events and keyframe feedback; restore only after a smoke pass |
| TTS stalls/underruns | Cancel at a safe boundary, stop old epoch, provide a counted failure/status | Check horizon/service/CPU traces; do not replay a partial segment automatically |
| GPU OOM or driver fault | Mark worker unready, stop affected media, release/fence leases | Restart isolated worker after diagnosis; lower admitted profile; retain memory trace |
| Network/ICE failure | Offer reconnect/text path; bounded session grace | Validate candidates/TURN/certificates; fresh mapping after reconnect |
| Database unavailable | Reject new sessions/jobs; existing sessions continue only while ownership/authorization leases remain safely valid | Restore DB, reconcile reservations and revocation tombstones before admission |
| Object store unavailable | Keep pinned lawful assets serving only under valid leases; reject loads/enrollment | Verify object integrity before reloading; no empty-asset fallback labeled ready |
| Bad model/asset rollout | Stop new assignment to bad revision | Route new sessions to last known good profile; drain existing unless unsafe; preserve failure report |
| Credential leak/asset revocation | Revoke application access, stop source publishers, close room generations and deny grants | Rotate credentials, reconcile rejoin attempts, purge scoped caches and document residual access windows |
| Full disk | Stop intake/provisioning; preserve audit/state | Clean only TTL-expired staging/cache under ownership rules; never delete active artifacts blindly |

## 7. Rollout, rollback and backup

Use expand/contract database migrations: new code first understands old/new fields, backfill asynchronously, and remove old fields only after the rollback window. A release declares compatible schema ranges. Rollback is a tested application/profile switch; do not automatically run destructive down migrations against production data.

Prepare and warm the new version on spare capacity, run synthetic and consented canaries, then route new sessions through a small canary cohort. Compare first response, quality sentinels and errors before wider admission. Old workers drain; maintain the old image/model/asset manifests until the rollback window closes. Single-GPU local deployments without spare memory use an announced stop/start maintenance path, not a false zero-downtime claim.

Drain order: remove from admission/readiness → stop new reservations → allow existing sessions to finish within the configured maximum/grace → warn remaining clients → cancel/close remaining sessions and flush resources → terminate. Match orchestrator termination grace to this policy; a preStop hook alone does not extend an exhausted termination deadline. Never replace a model under an active session without a tested explicit transition.

Back up metadata, approved asset manifests, required lawful artifacts and consent/deletion records with encryption and restricted restore access. Verify restore into an isolated environment at each release and periodically thereafter; compare object hashes and replay revocation tombstones before serving. Do not restore public traffic or provider credentials as a side effect of a test. State whether original media is retained, its retention window and how deleted subjects are excluded from restored service.

## 8. CI/CD and release evidence

Required pipeline tiers:

1. **Every relevant PR:** lint/types/schema compatibility, CPU contract tests, API/database tenancy/idempotency tests, offline deterministic media and config validation. Build browser assets; scan secrets and dependency/license changes.
2. **Integration:** real PostgreSQL and LiveKit with synthetic fixture; browser UI start/permission/failure/control flow; enrollment retry/cancel/revoke; mocked providers only in this tier and labeled as such.
3. **Provisioned GPU:** actual pinned models/checkpoints, native SDK/encoder parity, first-horizon/retarget quality, memory, lifetimes and repeated session churn. Network downloads excluded from test execution.
4. **Staging acceptance:** real browser/microphone path, external network/forced TURN, loudspeaker AEC, interruption-resume, real enrollment/approval/revocation, admitted N and eight-hour soak, worker failure, drain and upgrade/rollback.
5. **Release:** SBOM/licenses and consent scope, signed artifact manifests/checksums, compatibility matrix, protected quality report, performance raw records, backup restore receipt and operator runbooks.

Headless browser checks do not replace physical audio/device validation. A passing mock stack cannot satisfy the real-model release. Publish supported profiles with benchmark hardware, Q0 evidence and actual terms. Research-deployable requires assets, quality, cancel-resume and restore to pass within research access conditions; commercial-ready additionally requires commercial rights. Deferred commercial clearance alone does not fail research deployment.
