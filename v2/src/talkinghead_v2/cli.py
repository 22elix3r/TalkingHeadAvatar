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
    tts = commands.add_parser("tts", help="explicit real-model Qwen qualification")
    tts_actions = tts.add_subparsers(dest="tts_action", required=True)
    benchmark = tts_actions.add_parser("benchmark", help="real native callback cohort/soak; no device presentation")
    for name in ("runtime", "texts", "report", "log"):
        benchmark.add_argument(f"--{name}", type=Path, required=True)
    benchmark.add_argument("--seconds", type=float, default=0)
    benchmark.add_argument("--save-waves", type=Path)
    speak = tts_actions.add_parser("speak", help="real Qwen/P01 local playback and output-monitor evidence")
    for name in ("runtime", "report", "log", "capture"):
        speak.add_argument(f"--{name}", type=Path, required=True)
    speak.add_argument("--text", required=True)
    speak.add_argument("--device")
    speak.add_argument("--interrupt-after", type=float)
    speak.add_argument("--resume-text", default="This is the new generation after the interruption.")
    audio = commands.add_parser("audio", help="explicit local output-device qualification")
    audio_actions = audio.add_subparsers(dest="audio_action", required=True)
    calibrate = audio_actions.add_parser("calibrate", help="play two calibration tones and capture output monitor")
    calibrate.add_argument("--report", type=Path, required=True)
    calibrate.add_argument("--capture", type=Path, required=True)
    calibrate.add_argument("--device", help="PortAudio device name; default uses the system output")
    args = parser.parse_args(argv)
    if args.command == "audio":
        import asyncio
        from talkinghead_v2.audio.qualify import run

        result = asyncio.run(run(args))
        args.report.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({"passed": result["passed"], "analysis": result["analysis"]}))
        return 0 if result["passed"] else 1
    if args.command == "tts":
        import asyncio
        import math
        if args.tts_action == "speak":
            from talkinghead_v2.providers.tts.speech import run

            if args.interrupt_after is not None and (not math.isfinite(args.interrupt_after) or args.interrupt_after <= 0):
                parser.error("interrupt-after must be finite and positive")
            result = asyncio.run(run(args))
            print(json.dumps({"status": result["status"], "actual_device_playback": True, "turns": len(result["turns"])}))
            return 0
        from talkinghead_v2.providers.tts.benchmark import run

        if not math.isfinite(args.seconds) or args.seconds < 0:
            parser.error("seconds must be finite and nonnegative")
        result = asyncio.run(run(args))
        print(json.dumps({"completed_requests": result["completed_requests"],
                          "failed_requests": len(result["failed_requests"]),
                          "stability_30_minutes_passed": result["stability_30_minutes_passed"]}))
        return 1 if result["failed_requests"] else 0
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
