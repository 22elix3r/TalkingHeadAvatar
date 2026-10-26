# TalkingHeadAvatar V2 foundation

P00 supplies an isolated Python package, strict local TOML config and read-only host diagnostics. P01 adds typed media contracts, exact sample accounting, stateful reference resampling, bounded queues, epoch cancellation and synthetic PCM replay. P02 is in progress: explicit Qwen provisioning, official offline reference, native CUDA/F16 callbacks and a local callback-driven output path. V1 remains separate. See the [phase guide](../docs/v2/IMPLEMENTATION_PHASES.md), [implementation status](../docs/v2/IMPLEMENTATION_STATUS.md) and [P02 setup/evidence](docs/p02.md).

From the repository root, with `uv` and Python 3.12 already installed:

```sh
cd v2
uv sync --locked --python /usr/bin/python3.12
.venv/bin/tha --help
.venv/bin/tha doctor --profile local --config configs/local/doctor.toml
.venv/bin/tha doctor --profile local --config configs/local/doctor.toml --json
.venv/bin/tha replay
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q
```

`uv sync` installs this package editable into **`v2/.venv`**, with no system-site packages. It uses the committed `uv.lock`; first setup may fetch Python build/test packages from PyPI, never model weights. Use an existing Python 3.12 path on other hosts. Python 3.12 is the validated application series; other series are intentionally excluded until qualified. Keep the repository-root V1 `.venv` separate. After setup, help, doctor, imports and tests work offline. `uv sync --locked --offline --python /usr/bin/python3.12` can reconstruct the environment when build/test dependencies are already cached locally.

The core has **zero runtime dependencies**. Only the dev dependency group adds pytest. The build backend is separately pinned in `pyproject.toml`. Model/inference dependencies will be provisioned explicitly in P02 with their own compatibility evidence; neither torch nor any inference runtime belongs in the CPU foundation. No empty inference adapters or untested inference version pins are supplied here.

`tha` is also available as `.venv/bin/python -m talkinghead_v2`. Use `tha doctor --help` for flags. Doctor exit codes: **0** for a successful preflight, including optional-tool/cache warnings; **1** for diagnostic errors such as insufficient output space; **2** for invalid arguments/config. A zero exit code does not establish model or deployment readiness. Doctor JSON includes `schema_version`, `profile`, `status` and named checks with `status`/`detail`.

The [example config](configs/local/doctor.toml) points to `v2/var/` and requires 1 GiB of free output space. Keys are `profile` (only `local`), `artifact_cache_dir`, `output_dir` and nonnegative integer `min_free_bytes`. Unknown keys, wrong types and malformed TOML fail clearly. Relative paths are based on the config file's directory; `~` expands, but `$VARIABLE` interpolation is not supported. Omitted keys use defaults: artifact cache under `${XDG_CACHE_HOME:-~/.cache}/talkinghead-v2/artifacts`, output under the caller's current directory at `.tha-v2/output`, and 1 GiB minimum free space. Supplying a missing config file fails instead of silently using defaults.

Doctor queries Python/virtual-environment identity, `nvidia-smi` device/driver information, `nvcc --version` and FFmpeg's compiled NVENC encoder list. Each optional subprocess has a three-second timeout. These queries do not allocate CUDA tensors or run kernels/encoding. Missing, failing or hung GPU tools produce warnings. Tool/compiler presence does not establish CUDA runtime compatibility or usable encoding. Native media qualification belongs to later phases.

Doctor inspects only the configured artifact cache's immediate directory entries. A nonempty directory is **presence only**, not a validated Qwen cache; no weight, biometric file, checkpoint or pickle is opened. Existing V1/Hugging Face caches are not searched or migrated. P02 will add verified model/tokenizer/runtime identities, hashes, rights and an authorized reference voice. Qwen3-TTS Base 1.7B remains the selected provider, X-Vector first; it is not provisioned by P00.

Output checks use the nearest existing ancestor, permission bits and filesystem free space. Doctor never creates cache/output directories or writes a test file; ACLs, quotas and actual writes must be qualified when output is introduced. Diagnostics include local paths and device descriptions, so review them before sharing. The [baseline](docs/baseline.md) records P00 setup evidence and inherited V1 limitations.

Run the deterministic P01 fixture with `.venv/bin/tha replay --json`. By default it writes no artifact; `--report /tmp/tha-p01.json` explicitly saves a JSON report and creates its parent directory if needed. Replay returns 0 when all checks pass and a nonzero status on failure. Its procedural PCM includes irregular/empty chunks, silence, 24/44.1 kHz inputs, 48 kHz output, a partial final chunk, blocked-producer cancellation and late callbacks around restart. The sink records a synthetic audio schedule; it never plays audio or acknowledges a device/browser buffer. No conversation-latency, audiovisual-quality or real-time resampler claim follows from this fixture. See the [P01 contract and evidence notes](docs/p01.md).

P02 inference dependencies live in the separately locked `runtimes/qwen-reference/` environment. CPU tests still require no torch or model weights. `tha artifacts provision` explicitly downloads the selected pinned Base/tokenizer files; serving verification does not download. `tha tts benchmark`, `tha tts speak` and `tha audio calibrate` perform deliberate real-model/device actions described in the P02 notes. Approved human reference/listening and sustained native reliability qualification remain open; P02 has not been marked complete.
