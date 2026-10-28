# P06 offline API software image

Local packaging checkpoint, 2026-10-29. An explicit builder now creates a
private scratch image from the already verified Fedora/Python/API environment,
first-party serving code, resolved software dependencies and installed driver
libraries. No base image pull, package installation, model download, daemon
reconfiguration or host-driver replacement occurs.

The candidate passed its serving entry-point help, API dependency imports and
read-only NVIDIA driver inspection. It has **not** yet qualified model/provider
startup, native CUDA inference, browser playback, or container lifecycle/recovery.
Those remain P06 work. The [sustained actor cohort](p06-resource-stability.md)
also has a separate motion-underrun failure; this image does not resolve it.

## Actual scope and provenance

The builder audits the existing shared API environment against its recorded
installed-file fingerprint, reuses the existing API lock and writes a per-file
software inventory. It stages the selected software roots only, refuses cache/
profile paths and destination escapes, and verifies each copied file's identity
and extent. The environment's transient `.lock` is omitted without changing it.
Python caches and development include directories are excluded. Selected Qwen,
ASR and Claire/native caches, web build and mutable profile/database state stay
external; installed dependency data remains part of the software environment.

The installed driver package inventory supplies late-loaded libraries, so
dependency selection no longer depends on observing a currently active GPU
process. `ldd` inspection includes the ELF interpreter and rejects missing
dependencies. The generated Dockerfile uses `FROM scratch`, the observed
unprivileged UID/GID **1000:1000**, offline flags and the existing local serving
entry point. This is a host-qualified Linux ABI/driver image, not a general
cross-platform release.

Actual candidate:

- Image ID `sha256:917fe97b846d46e34aad0499df33d85b28e6b5a5f905f50af1f92a836e1bfdae`.
- Docker-reported size **1,928,851,885 bytes**; **5,309** inventory records.
- Logical inventory bytes **4,353,346,965**, including path aliases; this is
  not a measurement of unique physical storage or runtime RAM.
- Software inventory SHA-256
  `5eb77f22c8ce3759fb1e3d34598aaf6705e1c79946cec6699d21c5ede212b76e`.

Source API files were unchanged by packaging. Builder/stager source copies,
dependency inspection, inventory and build logs remain in the private context.
Installed wheel/license data and host package terms remain separate from the
first-party repository license. The private combined image has no asserted
redistribution clearance; it was not pushed to a registry.

## Actual commands and checks

With the existing qualified profile/cache/environment, from the repository root:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/.venv/bin/python \
  v2/scripts/build_p06_host_image.py --profile p06-publisher \
  --image talkinghead-v2-api:p06-host-20261029-4 \
  --directory v2/var/p06-image/host-image-20261029-4 \
  --report v2/var/p06-image/host-image-20261029-4.json
```

Use fresh paths/tag for reruns: existing evidence and image tags are preserved.
The build uses `docker build --network=none --pull=false`. Its automatic
`--help` check runs with no network, read-only rootfs, all capabilities dropped
and `no-new-privileges`. A separate no-network/no-device/no-cache import check
passed NumPy, PyAV, CTranslate2, Faster Whisper, SQLAlchemy, LiveKit and the
serving application. These are import/ABI checks, not inference.

The host has no configured NVIDIA container runtime. A separate ephemeral
inspection container used only explicit existing NVIDIA device nodes, with the
same read-only rootfs, no network and dropped capabilities. Existing
`/usr/bin/nvidia-smi` reported **RTX 4070 Ti SUPER, 580.178.04, 16,376 MiB** and
exited 0. NVML inspection is not proof of CUDA kernels, NVENC, actual provider
operation or user playback inside this image. No daemon or device configuration
was changed.

Earlier attempts are retained: staging rejected the transient world-writable
environment lock; another attempt exposed a builder-local variable collision
while saving source provenance. Both were corrected before the final build.
Tests cover selected-root/cache exclusion, source/destination symlink escapes,
dependency resolution and lock preservation. Final core command from `v2/`,
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q --junitxml=var/p06-image/host-image-core-tests.xml`,
passed **437 tests in 7.88 s**. No new serving API-suite result is claimed.

[Sanitized evidence](../benchmarks/reports/p06-api-host-image.json) binds the
image/software/source/test identities and inspection receipts without profile
credentials, model contents or private filesystem inventories. Next: explicit
cache/device/profile mounting, actual image provider startup and browser
recovery, then owned image/Compose management. No such future commands are
claimed implemented here. P06 remains in progress.
