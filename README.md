<p align="center">
  <img src="docs/assets/talkingheadavatar-wordmark.svg" alt="TalkingHeadAvatar" width="720">
</p>

<p align="center">
  <a href="docs/v2/IMPLEMENTATION_STATUS.md"><img src="https://img.shields.io/badge/status-research-ff5a1f" alt="Status: research"></a>
  <a href="v2/pyproject.toml"><img src="https://img.shields.io/badge/python-3.12-3776ab" alt="Python 3.12"></a>
  <a href="v2/docs/p02.md"><img src="https://img.shields.io/badge/speech-Qwen3--TTS-2d8c83" alt="Speech: Qwen3-TTS"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-497cbd" alt="License: Apache 2.0"></a>
</p>

<p align="center">
  <a href="#getting-started">Getting started</a> &nbsp; | &nbsp;
  <a href="#how-it-works">How it works</a> &nbsp; | &nbsp;
  <a href="docs/v2/IMPLEMENTATION_STATUS.md">Project status</a> &nbsp; | &nbsp;
  <a href="https://github.com/22elix3r/TalkingHeadAvatar/tree/v1/legacy">V1 archive</a>
</p>

---

<p align="center"><strong>TalkingHeadAvatar is a local-first research stack for spoken conversation and animated avatars.</strong></p>

The project brings speech, conversation, facial motion and real-time media together in a modular V2 runtime. Its first milestone is a consented voice and avatar that can speak, move and respond to interruption on a shared audio timeline.

This is an active research implementation. Local browser speech and parts of the motion and rendering pipeline have been exercised, while complete animated browser delivery, visual quality, physical audio/video synchronization and release readiness remain open. The [implementation status](docs/v2/IMPLEMENTATION_STATUS.md) records what has actually been tested and what comes next.

## How it works

V2 separates speech recognition, language, speech generation and motion into replaceable providers. A session actor manages turns and cancellation, while audio samples provide the clock for motion and presentation. The initial speech provider is **Qwen3-TTS Base 1.7B**, using X-Vector voice conditioning first.

The local browser reference uses self-hosted LiveKit for media. A separate research path exercises NVIDIA Claire facial motion with native CUDA rendering and NVENC; connecting that path to the full browser experience is still in progress. Asset compatibility, consent and provenance are checked separately from model quality.

## Getting started

The V2 foundation uses an isolated Python 3.12 environment. With [uv](https://docs.astral.sh/uv/) and Python 3.12 installed, run from the repository root:

```sh
cd v2
uv sync --locked --python /usr/bin/python3.12
.venv/bin/tha --help
.venv/bin/tha doctor --profile local --config configs/local/doctor.toml
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q
```

Use your Python 3.12 path if it differs. The first environment setup may fetch Python packages; the core tests run offline without model weights. Running inference or the browser reference requires separately provisioned runtimes and authorized assets. Model provisioning is explicit; follow the [V2 setup](v2/README.md), [browser guide](v2/docs/p04.md) and [local profiles](v2/docs/p06-local.md) for their requirements.

## Project status

The local reference has exercised streamed Qwen speech and browser audio, static avatar import and revocation, and native media and Claire rendering. Live animated browser integration, independent quality evaluation, human enrollment, physical A/V measurement and release gates remain in progress. Results from local experiments do not establish a product latency or quality guarantee.

`main` contains V2. The original V1 runtime and history remain available on [`v1/legacy`](https://github.com/22elix3r/TalkingHeadAvatar/tree/v1/legacy); see the [V1 audit](docs/v2/V1_AUDIT.md) for its known limitations.

## Documentation

| Guide | What it covers |
| --- | --- |
| [Implementation status](docs/v2/IMPLEMENTATION_STATUS.md) | Current phases, evidence, limitations and next work |
| [Implementation phases](docs/v2/IMPLEMENTATION_PHASES.md) | Milestones and acceptance gates |
| [V2 build plan](V2_RESEARCH_AND_BUILD_PLAN.md) | Architecture and research direction |
| [V2 setup](v2/README.md) | Environments, commands and local runtime boundaries |
| [Browser reference](v2/docs/p04.md) | Local conversation setup and media behavior |
| [Validation protocol](docs/v2/VALIDATION_PROTOCOL.md) | Measurement and quality-evaluation boundaries |
| [Research evidence](docs/v2/RESEARCH_EVIDENCE.md) | Component sources, evidence and separately recorded terms |
| [Repository history](docs/v2/REPOSITORY_HISTORY.md) | V1 preservation and scoped V2 attribution repair |

## License

First-party V2 software and documentation in this repository are available under the [Apache License 2.0](LICENSE). Third-party code, models and weights, datasets, voices, likenesses and other assets retain their own licenses and permissions. The license does not change the terms of the separately preserved V1 history. Consult the [research evidence](docs/v2/RESEARCH_EVIDENCE.md) and individual component notices before use or redistribution.
