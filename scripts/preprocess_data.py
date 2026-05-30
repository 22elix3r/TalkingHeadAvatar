#!/usr/bin/env python3
"""
Data Preprocessing Helper — Phase 2
=====================================
Validates and prepares the subject dataset structure required by GaussianAvatars.

Checks:
  1. VHAP output: canonical_flame_param.npz + per-frame flame_param/*.npz
  2. Transform JSONs (train / val / test)
  3. Images and masks directories
  4. Audio extraction from raw video (if ffmpeg is available)
  5. Generates a persona.json template if missing

Usage:
    source activate_env.sh
    python scripts/preprocess_data.py --subject MY_SUBJECT [--video path/to/video.mp4]

Output:
    • data/{subject}/audio_full.wav         (extracted from video if --video given)
    • data/{subject}/voice_reference.wav    (30-45s segment for TTS cloning)
    • data/{subject}/persona.json           (template if absent)
    • Summary report of dataset completeness
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

REQUIRED_FLAME_KEYS = {"shape", "expr", "rotation", "neck_pose", "jaw_pose",
                        "eyes_pose", "translation"}

PERSONA_TEMPLATE = {
    "name": "YOUR_NAME",
    "role": "Your Title, Your Company",
    "communication_style": (
        "Describe your communication style here: e.g., direct, data-driven, "
        "prefers concise answers."
    ),
    "vocabulary_patterns": ["phrase_1", "phrase_2", "phrase_3"],
    "domain_expertise": ["domain_1", "domain_2"],
    "known_opinions": {
        "topic_1": "Your position on topic 1.",
        "topic_2": "Your position on topic 2.",
    },
    "biographical_context": (
        "A few sentences about background: years of experience, notable roles, location."
    ),
    "tone_guardrails": (
        "E.g., professional, direct, avoids filler words, acknowledges others before disagreeing."
    ),
}


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


# ──────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────

def check_vhap_output(subject_dir: Path) -> dict[str, bool | int | str]:
    """Validate the transformsVHAP/ directory produced by VHAP."""
    report: dict[str, bool | int | str] = {}
    transforms_dir = subject_dir / "transformsVHAP"

    report["transformsVHAP_exists"] = transforms_dir.exists()
    if not transforms_dir.exists():
        return report

    canonical = transforms_dir / "canonical_flame_param.npz"
    report["canonical_flame_param"] = canonical.exists()

    flame_param_dir = transforms_dir / "flame_param"
    report["flame_param_dir"] = flame_param_dir.exists()

    if flame_param_dir.exists():
        npz_files = sorted(flame_param_dir.glob("*.npz"))
        report["flame_param_count"] = len(npz_files)

        if npz_files:
            try:
                import numpy as np
                sample = np.load(str(npz_files[0]))
                missing = REQUIRED_FLAME_KEYS - set(sample.keys())
                report["flame_param_keys_ok"] = len(missing) == 0
                if missing:
                    report["flame_param_missing_keys"] = str(missing)
            except Exception as exc:
                report["flame_param_load_error"] = str(exc)
    else:
        report["flame_param_count"] = 0

    for split in ("train", "val", "test"):
        json_path = transforms_dir / f"transforms_{split}.json"
        report[f"transforms_{split}_json"] = json_path.exists()

    return report


def check_images_and_masks(subject_dir: Path) -> dict[str, bool | int]:
    """Check view_000/images/ and view_000/masks/."""
    report: dict[str, bool | int] = {}
    view_dir = subject_dir / "view_000"
    report["view_000_exists"] = view_dir.exists()

    if not view_dir.exists():
        return report

    images_dir = view_dir / "images"
    masks_dir = view_dir / "masks"

    report["images_dir_exists"] = images_dir.exists()
    report["masks_dir_exists"] = masks_dir.exists()

    if images_dir.exists():
        imgs = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
        report["image_count"] = len(imgs)

    if masks_dir.exists():
        masks = list(masks_dir.glob("*.png"))
        report["mask_count"] = len(masks)

    return report


def check_audio(subject_dir: Path) -> dict[str, bool]:
    report: dict[str, bool] = {}
    report["audio_full_wav"] = (subject_dir / "audio_full.wav").exists()
    report["voice_reference_wav"] = (subject_dir / "voice_reference.wav").exists()
    report["persona_json"] = (subject_dir / "persona.json").exists()
    return report


# ──────────────────────────────────────────────────────────
# Audio extraction
# ──────────────────────────────────────────────────────────

def extract_audio(video_path: Path, subject_dir: Path, voice_ref_start: str = "00:00:30",
                  voice_ref_duration: str = "00:00:45") -> None:
    """Extract full audio and a voice reference segment from a video file."""
    if not _ffmpeg_available():
        print("[preprocess] WARNING: ffmpeg not found — skipping audio extraction.")
        return

    audio_full = subject_dir / "audio_full.wav"
    voice_ref = subject_dir / "voice_reference.wav"

    if not audio_full.exists():
        print(f"[preprocess] Extracting audio from {video_path} …")
        result = _run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_full),
        ], check=False)
        if result.returncode != 0:
            print(f"[preprocess] ERROR: ffmpeg failed:\n{result.stderr}")
            return
        print(f"[preprocess] ✓ audio_full.wav saved ({audio_full.stat().st_size // 1024} KB)")
    else:
        print(f"[preprocess] audio_full.wav already exists — skipping.")

    if not voice_ref.exists():
        print(f"[preprocess] Extracting voice reference segment …")
        result = _run([
            "ffmpeg", "-y", "-i", str(audio_full),
            "-ss", voice_ref_start, "-t", voice_ref_duration,
            str(voice_ref),
        ], check=False)
        if result.returncode != 0:
            print(f"[preprocess] ERROR: ffmpeg failed for reference clip:\n{result.stderr}")
            return
        print(f"[preprocess] ✓ voice_reference.wav saved ({voice_ref.stat().st_size // 1024} KB)")
    else:
        print(f"[preprocess] voice_reference.wav already exists — skipping.")


# ──────────────────────────────────────────────────────────
# Persona template
# ──────────────────────────────────────────────────────────

def ensure_persona_template(subject_dir: Path) -> None:
    persona_path = subject_dir / "persona.json"
    if persona_path.exists():
        print(f"[preprocess] persona.json already exists — skipping.")
        return

    persona_path.write_text(json.dumps(PERSONA_TEMPLATE, indent=2))
    print(f"[preprocess] ✓ Created persona.json template at {persona_path}")
    print(f"[preprocess]   → Edit this file before running run_demo.py")


# ──────────────────────────────────────────────────────────
# Report printer
# ──────────────────────────────────────────────────────────

def _ok(val) -> str:
    if isinstance(val, bool):
        return "✅" if val else "❌"
    return str(val)


def print_report(subject: str, vhap: dict, media: dict, audio: dict) -> bool:
    print(f"\n{'='*60}")
    print(f"  Dataset report for subject: {subject}")
    print(f"{'='*60}")

    all_ok = True

    print("\n[VHAP output]")
    for k, v in vhap.items():
        ok = v if isinstance(v, bool) else True
        if isinstance(v, bool) and not v:
            all_ok = False
        print(f"  {_ok(v):3} {k}")

    print("\n[Images & masks]")
    for k, v in media.items():
        ok = v if isinstance(v, bool) else True
        if isinstance(v, bool) and not v:
            all_ok = False
        print(f"  {_ok(v):3} {k}")

    print("\n[Audio & persona]")
    for k, v in audio.items():
        ok = v if isinstance(v, bool) else True
        if isinstance(v, bool) and not v:
            all_ok = False
        print(f"  {_ok(v):3} {k}")

    print(f"\n{'='*60}")
    if all_ok:
        print("  ✅ Dataset is COMPLETE — ready for GaussianAvatars training.")
    else:
        print("  ⚠️  Dataset is INCOMPLETE — fix the ❌ items above before training.")
    print(f"{'='*60}\n")

    return all_ok


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate and prepare subject data for TalkingHeadAvatar training."
    )
    parser.add_argument(
        "--subject", required=True,
        help="Subject directory name under data/ (e.g. MY_SUBJECT)",
    )
    parser.add_argument(
        "--video", default=None,
        help="Optional path to raw video (.mp4). If given, audio is extracted automatically.",
    )
    parser.add_argument(
        "--voice_ref_start", default="00:00:30",
        help="Start timestamp for voice reference clip (default: 00:00:30)",
    )
    parser.add_argument(
        "--voice_ref_duration", default="00:00:45",
        help="Duration of voice reference clip in HH:MM:SS (default: 00:00:45)",
    )
    parser.add_argument(
        "--create_dirs", action="store_true",
        help="Create missing directory structure under data/{subject}/",
    )
    args = parser.parse_args()

    subject_dir = DATA_ROOT / args.subject
    subject_dir.mkdir(parents=True, exist_ok=True)

    if args.create_dirs:
        (subject_dir / "transformsVHAP" / "flame_param").mkdir(parents=True, exist_ok=True)
        (subject_dir / "view_000" / "images").mkdir(parents=True, exist_ok=True)
        (subject_dir / "view_000" / "masks").mkdir(parents=True, exist_ok=True)
        print(f"[preprocess] Created directory structure under {subject_dir}")

    # Extract audio from video
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"[preprocess] ERROR: video file not found: {video_path}")
            sys.exit(1)
        extract_audio(video_path, subject_dir, args.voice_ref_start, args.voice_ref_duration)

    # Ensure persona template
    ensure_persona_template(subject_dir)

    # Validate and report
    vhap_report = check_vhap_output(subject_dir)
    media_report = check_images_and_masks(subject_dir)
    audio_report = check_audio(subject_dir)

    ok = print_report(args.subject, vhap_report, media_report, audio_report)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
