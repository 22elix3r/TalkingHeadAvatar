# Pinned Python Claire media wrapper

Local functional observation, 2026-10-27. [Evidence](../benchmarks/reports/p06-claire-python.json) covers the actual Python process wrapper and native SFU sender, before actor/API integration. The normal preview is unchanged. The [native worker](p03-claire-preroll.md) has separate recorded/live receiver evidence; this wrapper cohort does not add a browser or physical A/V result.

`ClaireMediaConfig` checks a private operator descriptor, bounded file inventory, hashes, ownership/modes, source revision, engine/build provenance, source model dependencies, exact skin/tongue topology, explicit scene/camera and vendor research scope. It distinguishes vendor consent assertion from human enrollment. The pinned model descriptor cannot redirect the SDK to unlisted files. A separate explicit check rejects a changed GPU UUID/driver before loading an engine. `describe_claire_media.py` only pins already-built binaries with a complete live source-frame qualification; it never downloads/builds weights or modifies host software.

The actual runtime inventory contains **183 files**: model/engine/scene, isolated TensorRT, SDK, native/LiveKit libraries and resolved host support libraries. Inventory bounds were adjusted from the initial insufficient 128 limit to 256 after observing the installed dependency tree; the original failed private descriptor is preserved. File sizes are bounded individually and in aggregate; hashes are streamed. Root-owned host libraries remain read-only inputs. The first successful descriptor is `var/p06-claire/runtime-20261027-v2.json`, SHA-256 `a62ed9a9cce3d0c711588282c5df3526faaa79c1991c9be64b5b9a5b31a88a29`.

`ClaireMedia` reuses the tested native process lifecycle, readiness/EOF events, priority signal, subprocess-creation shielding and cleanup. Hooks leave static native behavior intact. It sends the paired incremental FIR protocol, checks exact valid/padding counts, queue bounds and source-frame conservation, stages no imported-identity substitution, and rejects a changed serving revision. A key from another session is rejected before it can cancel the current owner or import the optional media SDK. Child CUDA cache stays private under the profile; preload/audit overrides are removed only from the child environment.

Actual commands from the repository root:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/.venv/bin/python \
  v2/scripts/describe_claire_media.py \
  --engine v2/var/p03/claire-engine-fp32-n1-final \
  --scene v2/var/p03/claire-render-scene \
  --output v2/var/p06-claire/runtime-20261027-v2.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/runtimes/session-api/.venv/bin/python \
  v2/scripts/qualify_claire_media.py \
  --config v2/var/p04/test-runtime.json \
  --runtime v2/var/p06-claire/runtime-20261027-v2.json \
  --directory v2/var/p06-claire/python-wrapper-cache-qualified \
  --report v2/benchmarks/reports/p06-claire-python.json
cd v2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q \
  --junitxml=var/p06-claire/wrapper-core-tests.xml
cd runtimes/session-api
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q \
  --junitxml=../../var/p06-claire/wrapper-api-tests.xml
```

**328 core tests passed in 5.85 s; 24 isolated API/PostgreSQL tests passed in 3.64 s.** Added 20 model-free checks for scope/identity/dependency tampering, device validation, loss/queue reporting, blocked-pipe fencing and unchanged parent environment. The actual wrapper cohort passes **three epochs**: full recorded Qwen, cancellation with an actually backpressured producer, then clean full recorded Qwen. Both complete epochs conserve **133,290 valid samples + 1,110 padding**. All three publication pairs differ; retired private epoch directories are purged and device admission is reacquired. Blocked owner retirement is **533.38 ms**, not receiver-safe suppression or a ≤200 ms acceptance pass.

No download, global dependency/driver/config change, V1 mutation, training or cloud deployment occurred. Asset/voice inputs remain the source-matched vendor research mesh and synthetic voice reference. The wrapper is not selected by an API profile yet. Next: bind a distinct builtin research asset/revision, use an isolated local API/database/cookie scope, and validate real typed/microphone browser conversation and fresh-track cancellation. Human enrollment, Gaussian retarget, independent quality, physical A/V and release gates remain open.
