#!/usr/bin/env python3
"""
Prepare source videos for subject-specific avatar retraining.

This script owns the pre-VHAP media step:
  - create the subject directory scaffold;
  - create or update a trim manifest;
  - normalize enabled manifest segments into one master video;
  - optionally create a short pilot video from the master;
  - extract audio_full.wav and voice_reference.wav.

It does not run VHAP or GaussianAvatars training. Use
scripts/run_avatar_retrain_pipeline.sh for those stages after the master video
has been built.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

DEFAULT_VIDEO_FILTER = (
    "scale=1024:1024:force_original_aspect_ratio=increase,"
    "crop=1024:1024"
)


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=check, text=True)


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool not found on PATH: {name}")


def _subject_dir(subject: str) -> Path:
    return DATA_ROOT / subject


def _default_manifest(subject: str, videos: list[Path]) -> dict[str, Any]:
    raw_dir = _subject_dir(subject) / "raw"
    segments: list[dict[str, Any]] = []
    for index, video in enumerate(videos, start=1):
        video_path = video
        if not video_path.is_absolute():
            video_path = (PROJECT_ROOT / video_path).resolve()
        try:
            rel_source = video_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_source = video_path
        segments.append(
            {
                "id": f"clip_{index:03d}",
                "source": str(rel_source),
                "start": "00:00:00",
                "end": "",
                "enabled": True,
                "video_filter": "",
                "notes": (
                    "Set exact start/end. Disable this segment if hands cover "
                    "mouth/chin/cheeks/eyes, face leaves frame, or subtitles "
                    "remain inside the crop."
                ),
            }
        )

    if not segments:
        segments = [
            {
                "id": "clip_001",
                "source": f"data/{subject}/raw/video_01.mp4",
                "start": "00:00:00",
                "end": "",
                "enabled": True,
                "video_filter": "",
                "notes": "Replace with the first cleaned source video.",
            },
            {
                "id": "clip_002",
                "source": f"data/{subject}/raw/video_02.mp4",
                "start": "00:00:00",
                "end": "",
                "enabled": True,
                "video_filter": "",
                "notes": "Replace with the second cleaned source video.",
            },
        ]

    return {
        "subject": subject,
        "defaults": {
            "fps": 25,
            "size": 1024,
            "audio_sample_rate": 16000,
            "video_filter": DEFAULT_VIDEO_FILTER,
        },
        "quality_rules": [
            "Use one same-identity speaker only.",
            "Visible face and audible speech must be the same person.",
            "Reject spans where hands cover lips, jawline, chin, cheeks, nose, or eyes.",
            "Crop out baked captions, watermarks, and UI overlays before VHAP.",
            "Keep natural head motion; do not apply heavy face stabilization.",
        ],
        "segments": segments,
    }


def init_subject(
    subject: str,
    videos: list[Path],
    copy_videos: bool,
    overwrite_manifest: bool,
) -> Path:
    subject_dir = _subject_dir(subject)
    raw_dir = subject_dir / "raw"
    master_dir = subject_dir / "master"
    for path in (raw_dir, master_dir, subject_dir / "view_000"):
        path.mkdir(parents=True, exist_ok=True)

    local_videos: list[Path] = []
    if copy_videos:
        for index, src in enumerate(videos, start=1):
            src = src.expanduser().resolve()
            if not src.exists():
                raise FileNotFoundError(src)
            suffix = src.suffix.lower() or ".mp4"
            dst = raw_dir / f"video_{index:02d}{suffix}"
            if src != dst:
                shutil.copy2(src, dst)
            local_videos.append(dst)
    else:
        local_videos = videos

    manifest = raw_dir / "trim_manifest.json"
    if overwrite_manifest or not manifest.exists():
        manifest.write_text(
            json.dumps(_default_manifest(subject, local_videos), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[prepare] Wrote manifest template: {manifest}")
    else:
        print(f"[prepare] Manifest already exists: {manifest}")
    return manifest


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "segments" not in data or not isinstance(data["segments"], list):
        raise ValueError(f"Manifest must contain a 'segments' list: {path}")
    return data


def _resolve_source(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _enabled_segments(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in manifest["segments"] if s.get("enabled", True)]


def _normalize_segment(
    segment: dict[str, Any],
    out_path: Path,
    default_filter: str,
    fps: int,
    audio_sample_rate: int,
) -> None:
    source = _resolve_source(str(segment["source"]))
    if not source.exists():
        raise FileNotFoundError(f"Segment source not found: {source}")

    cmd = ["ffmpeg", "-y", "-hide_banner"]
    start = str(segment.get("start", "")).strip()
    end = str(segment.get("end", "")).strip()
    if start:
        cmd += ["-ss", start]
    if end:
        cmd += ["-to", end]
    video_filter = str(segment.get("video_filter") or default_filter)
    cmd += [
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        video_filter,
        "-r",
        str(fps),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-ac",
        "1",
        "-ar",
        str(audio_sample_rate),
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    _run(cmd)


def build_master(
    manifest_path: Path,
    *,
    build_pilot: bool,
    pilot_seconds: int,
    voice_ref_start: str,
    voice_ref_duration: str,
) -> Path:
    _require_tool("ffmpeg")
    manifest = _load_manifest(manifest_path)
    subject = str(manifest.get("subject") or manifest_path.parents[1].name)
    subject_dir = _subject_dir(subject)
    master_dir = subject_dir / "master"
    work_dir = master_dir / "_segments"
    master_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    defaults = manifest.get("defaults", {})
    fps = int(defaults.get("fps", 25))
    audio_sample_rate = int(defaults.get("audio_sample_rate", 16000))
    default_filter = str(defaults.get("video_filter") or DEFAULT_VIDEO_FILTER)

    segments = _enabled_segments(manifest)
    if not segments:
        raise RuntimeError(f"No enabled segments in {manifest_path}")

    normalized: list[Path] = []
    for index, segment in enumerate(segments, start=1):
        segment_id = str(segment.get("id") or f"clip_{index:03d}")
        out_path = work_dir / f"{index:03d}_{segment_id}.mp4"
        _normalize_segment(segment, out_path, default_filter, fps, audio_sample_rate)
        normalized.append(out_path)

    concat_file = work_dir / "concat.txt"
    concat_file.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in normalized),
        encoding="utf-8",
    )

    master_video = master_dir / "master_video.mp4"
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(master_video),
        ]
    )

    audio_full = subject_dir / "audio_full.wav"
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-i",
            str(master_video),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_full),
        ]
    )

    voice_ref = subject_dir / "voice_reference.wav"
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-ss",
            voice_ref_start,
            "-t",
            voice_ref_duration,
            "-i",
            str(audio_full),
            str(voice_ref),
        ]
    )

    if build_pilot:
        pilot_video = master_dir / "pilot_video.mp4"
        _run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-i",
                str(master_video),
                "-t",
                str(pilot_seconds),
                "-c",
                "copy",
                str(pilot_video),
            ]
        )
        print(f"[prepare] Pilot video: {pilot_video}")

    print(f"[prepare] Master video: {master_video}")
    print(f"[prepare] Audio: {audio_full}")
    print(f"[prepare] Voice reference: {voice_ref}")
    return master_video


def _ffprobe_json(path: Path) -> dict[str, Any]:
    _require_tool("ffprobe")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def validate(subject: str) -> bool:
    subject_dir = _subject_dir(subject)
    master_video = subject_dir / "master" / "master_video.mp4"
    pilot_video = subject_dir / "master" / "pilot_video.mp4"
    audio_full = subject_dir / "audio_full.wav"
    manifest = subject_dir / "raw" / "trim_manifest.json"

    ok = True
    for path in (manifest, master_video, audio_full):
        exists = path.exists()
        ok = ok and exists
        print(f"[validate] {'OK ' if exists else 'MISS'} {path}")

    for video in (master_video, pilot_video):
        if not video.exists():
            continue
        info = _ffprobe_json(video)
        vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        astreams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
        if not vstreams:
            print(f"[validate] MISS video stream: {video}")
            ok = False
            continue
        v = vstreams[0]
        duration = float(info.get("format", {}).get("duration", 0.0))
        print(
            "[validate] video "
            f"{video}: {v.get('width')}x{v.get('height')} "
            f"fps={v.get('avg_frame_rate')} duration={duration:.2f}s "
            f"audio_streams={len(astreams)}"
        )
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare normalized source media for avatar retraining."
    )
    parser.add_argument("--subject", default="new_speaker")
    parser.add_argument("--videos", nargs="*", type=Path, default=[])
    parser.add_argument("--copy-videos", action="store_true")
    parser.add_argument("--overwrite-manifest", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--build-master", action="store_true")
    parser.add_argument("--build-pilot", action="store_true")
    parser.add_argument("--pilot-seconds", type=int, default=300)
    parser.add_argument("--voice-ref-start", default="00:00:30")
    parser.add_argument("--voice-ref-duration", default="00:00:45")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest
    if manifest is None:
        manifest = _subject_dir(args.subject) / "raw" / "trim_manifest.json"
    if not manifest.is_absolute():
        manifest = (PROJECT_ROOT / manifest).resolve()

    if args.init:
        manifest = init_subject(
            args.subject,
            args.videos,
            args.copy_videos,
            args.overwrite_manifest,
        )

    if args.build_master:
        if not manifest.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest}. Run with --init first."
            )
        build_master(
            manifest,
            build_pilot=args.build_pilot,
            pilot_seconds=args.pilot_seconds,
            voice_ref_start=args.voice_ref_start,
            voice_ref_duration=args.voice_ref_duration,
        )

    if args.validate:
        if not validate(args.subject):
            return 1

    if not (args.init or args.build_master or args.validate):
        print("Nothing to do. Use --init, --build-master, or --validate.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
