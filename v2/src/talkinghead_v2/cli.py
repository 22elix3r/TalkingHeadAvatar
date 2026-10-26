"""Local diagnostics and deterministic CPU contract replay."""

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

from talkinghead_v2 import __version__
from talkinghead_v2.config import ConfigError, load_config
from talkinghead_v2.doctor import diagnose


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tha", description="TalkingHeadAvatar V2 foundation")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="read-only environment diagnostics; no model loading")
    doctor.add_argument("--profile", choices=("local",), default="local")
    doctor.add_argument("--config", type=Path, help="TOML config; relative paths use its directory")
    doctor.add_argument("--json", action="store_true", dest="as_json", help="emit structured diagnostics")
    replay = commands.add_parser("replay", help="run P01 synthetic PCM/cancel fixture; no device playback")
    replay.add_argument("--json", action="store_true", dest="as_json")
    replay.add_argument("--report", type=Path, help="explicitly write the synthetic JSON report here")
    artifacts = commands.add_parser("artifacts", help="explicit Qwen provisioning or offline cache verification")
    artifacts.add_argument("action", choices=("provision", "verify"))
    artifacts.add_argument("--manifest", type=Path, required=True)
    artifacts.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "artifacts":
        from talkinghead_v2.assets.artifacts import ArtifactError, provision, read_manifest, verify_snapshot

        try:
            manifest = read_manifest(args.manifest)
            target = provision(args.cache, manifest) if args.action == "provision" else verify_snapshot(args.cache, manifest)
        except (ArtifactError, OSError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps({"model_id": manifest["model_id"], "revision": manifest["revision"],
                          "snapshot": str(target), "status": "verified"}, indent=2))
        return 0
    if args.command == "replay":
        from talkinghead_v2.replay import run_replay

        report = run_replay()
        serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.report is not None:
            try:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(serialized)
            except OSError as exc:
                parser.error(f"cannot write replay report: {exc}")
        if args.as_json:
            print(serialized, end="")
        else:
            print("P01 synthetic scheduling replay: passed")
            print(f"Completed output samples: {report['completed']['output_samples']}; "
                  f"stale callbacks rejected: {report['stale_callbacks_rejected']}")
            print("Actual device playback, latency and audiovisual quality remain unqualified.")
        return 0
    try:
        config = load_config(args.config, profile=args.profile)
    except ConfigError as exc:
        parser.error(str(exc))
    report = diagnose(config)
    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"TalkingHeadAvatar V2 doctor: profile={report.profile}, status={report.status}")
        for check in report.checks:
            print(f"[{check.status.upper()}] {check.name}: {check.detail}")
        print("P00 preflight only; model, inference, playback and deployment readiness are unqualified.")
    return 1 if report.status == "error" else 0
