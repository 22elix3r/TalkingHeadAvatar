#!/usr/bin/env python3
"""
Prepare a downloaded demo character for this repository's data/layout conventions.

This script:
1. Extracts one character's avatar checkpoint tar into output/demo_character/extracted/{character}/
2. Extracts one character's dataset from data/demo_character/splattingavatar.tar into
   data/demo_character/extracted/{character}/
3. Creates data/{subject}/ with:
   - transformsVHAP -> extracted transformsMetracker
   - view_000 -> extracted view_000
   - persona.json (template)
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_TAR = PROJECT_ROOT / "data" / "demo_character" / "splattingavatar.tar"
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "output" / "demo_character" / "splattingavatar"
DEFAULT_WEIGHTS_EXTRACT_ROOT = PROJECT_ROOT / "output" / "demo_character" / "extracted"
DEFAULT_DATA_EXTRACT_ROOT = PROJECT_ROOT / "data" / "demo_character" / "extracted"


def _persona_template(character: str) -> dict:
    display_name = character.replace("_", " ").title()
    return {
        "name": display_name,
        "role": "Demo Character",
        "communication_style": "concise, clear, and conversational in technical discussions",
        "vocabulary_patterns": ["Let's break it down.", "The key point is this."],
        "domain_expertise": ["software engineering", "AI avatars"],
        "known_opinions": {
            "engineering_quality": "Prefer reliable, testable implementation over quick hacks.",
            "communication": "Direct communication with clear next actions.",
        },
        "biographical_context": (
            f"This is a local demo persona derived from the downloaded character '{character}'."
        ),
        "tone_guardrails": "professional, collaborative, and brief",
    }


def _is_junk(parts: tuple[str, ...]) -> bool:
    return any(part.startswith("._") or part == ".DS_Store" for part in parts)


def _safe_dest(root: Path, rel_parts: tuple[str, ...]) -> Path:
    dest = root.joinpath(*rel_parts)
    root_resolved = root.resolve()
    dest_resolved = dest.resolve()
    if root_resolved != dest_resolved and root_resolved not in dest_resolved.parents:
        raise ValueError(f"Blocked unsafe archive path: {dest}")
    return dest


def _extract_member(archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> None:
    if member.isdir():
        destination.mkdir(parents=True, exist_ok=True)
        return
    if not member.isfile():
        return
    source = archive.extractfile(member)
    if source is None:
        raise RuntimeError(f"Failed to read archive member: {member.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def _weight_characters(weights_dir: Path) -> list[str]:
    if not weights_dir.exists():
        return []
    return sorted(path.stem for path in weights_dir.glob("*.tar"))


def _extract_weights_character(
    character: str,
    weights_tar: Path,
    extract_root: Path,
    force: bool,
) -> Path:
    character_dir = extract_root / character
    if force and character_dir.exists():
        shutil.rmtree(character_dir)
    if character_dir.exists():
        return character_dir

    extract_root.mkdir(parents=True, exist_ok=True)
    extracted = False
    with tarfile.open(weights_tar, "r:*") as archive:
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != character or _is_junk(parts):
                continue
            destination = _safe_dest(extract_root, parts)
            _extract_member(archive, member, destination)
            extracted = True
    if not extracted:
        raise RuntimeError(f"No '{character}' entries found in {weights_tar}")
    return character_dir


def _extract_dataset_character(
    character: str,
    dataset_tar: Path,
    extract_root: Path,
    force: bool,
) -> Path:
    character_dir = extract_root / character
    if force and character_dir.exists():
        shutil.rmtree(character_dir)
    if character_dir.exists():
        return character_dir

    extract_root.mkdir(parents=True, exist_ok=True)
    extracted = False
    with tarfile.open(dataset_tar, "r:*") as archive:
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if not parts or _is_junk(parts):
                continue
            rel_parts: tuple[str, ...] | None = None
            if parts[0] == character:
                rel_parts = parts
            elif len(parts) >= 2 and parts[1] == character:
                rel_parts = parts[1:]
            if rel_parts is None:
                continue
            destination = _safe_dest(extract_root, rel_parts)
            _extract_member(archive, member, destination)
            extracted = True
    if not extracted:
        raise RuntimeError(f"No '{character}' entries found in {dataset_tar}")
    return character_dir


def _replace_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _symlink(src: Path, dst: Path, force: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        _replace_path(dst)
    dst.symlink_to(src)


def _ensure_subject_layout(character: str, dataset_dir: Path, subject_dir: Path, force: bool) -> Path:
    transforms_src = dataset_dir / "transformsMetracker"
    view_src = dataset_dir / "view_000"
    if not transforms_src.exists():
        raise FileNotFoundError(f"Missing transformsMetracker in dataset: {transforms_src}")
    if not view_src.exists():
        raise FileNotFoundError(f"Missing view_000 in dataset: {view_src}")

    subject_dir.mkdir(parents=True, exist_ok=True)
    _symlink(transforms_src, subject_dir / "transformsVHAP", force=force)
    _symlink(view_src, subject_dir / "view_000", force=force)

    persona_path = subject_dir / "persona.json"
    if force or not persona_path.exists():
        persona_path.write_text(
            json.dumps(_persona_template(character), indent=2) + "\n",
            encoding="utf-8",
        )
    return persona_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one downloaded demo character for local use.")
    parser.add_argument(
        "--character",
        default=None,
        help="Character ID to prepare (e.g. biden_001). Defaults to first available weight tar.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Target subject directory under data/. Default: demo_{character}",
    )
    parser.add_argument("--weights_dir", default=str(DEFAULT_WEIGHTS_DIR))
    parser.add_argument("--dataset_tar", default=str(DEFAULT_DATASET_TAR))
    parser.add_argument("--weights_extract_root", default=str(DEFAULT_WEIGHTS_EXTRACT_ROOT))
    parser.add_argument("--data_extract_root", default=str(DEFAULT_DATA_EXTRACT_ROOT))
    parser.add_argument("--force", action="store_true", help="Re-extract and replace existing links/files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_dir = Path(args.weights_dir).resolve()
    dataset_tar = Path(args.dataset_tar).resolve()
    weights_extract_root = Path(args.weights_extract_root).resolve()
    data_extract_root = Path(args.data_extract_root).resolve()

    available = _weight_characters(weights_dir)
    if not available:
        raise FileNotFoundError(f"No weight tar files found in: {weights_dir}")

    character = args.character or available[0]
    if character == "sam_altman":
        raise ValueError("sam_altman is explicitly excluded; choose another demo character.")
    if character not in available:
        raise ValueError(f"Character '{character}' not found. Available: {', '.join(available)}")

    weights_tar = weights_dir / f"{character}.tar"
    if not dataset_tar.exists():
        raise FileNotFoundError(f"Dataset tar not found: {dataset_tar}")

    subject_name = args.subject or f"demo_{character}"
    subject_dir = (PROJECT_ROOT / "data" / subject_name).resolve()

    weights_character_dir = _extract_weights_character(
        character=character,
        weights_tar=weights_tar,
        extract_root=weights_extract_root,
        force=args.force,
    )
    dataset_character_dir = _extract_dataset_character(
        character=character,
        dataset_tar=dataset_tar,
        extract_root=data_extract_root,
        force=args.force,
    )
    persona_path = _ensure_subject_layout(
        character=character,
        dataset_dir=dataset_character_dir,
        subject_dir=subject_dir,
        force=args.force,
    )

    print("============================================================")
    print("Demo character prepared")
    print("============================================================")
    print(f"Character:      {character}")
    print(f"Subject:        {subject_name}")
    print(f"Avatar ckpt:    {weights_character_dir}")
    print(f"Dataset source: {dataset_character_dir}")
    print(f"Persona file:   {persona_path}")
    print("")
    point_path = weights_character_dir / "point_cloud" / "iteration_180000" / "point_cloud.ply"
    if point_path.exists():
        print("Local GaussianAvatars viewer:")
        print(
            "python GaussianAvatars/local_viewer.py "
            f"--point_path {point_path}"
        )
        print("")
    print("Project demo entrypoint:")
    print("python run_demo.py \\")
    print(f"  --avatar_ckpt {weights_character_dir} \\")
    print(f"  --persona_file {persona_path} \\")
    print("  --skip_gemma --skip_audio_listener --enable_stdin_fallback --dry_run")
    print("")
    print("Virtual-camera-only smoke test (OBS wiring):")
    print("python run_demo.py \\")
    print(f"  --avatar_ckpt {weights_character_dir} \\")
    print(f"  --persona_file {persona_path} \\")
    print("  --camera_device /dev/video10 --skip_gemma --skip_audio_listener --video_only")
    print("============================================================")


if __name__ == "__main__":
    main()
