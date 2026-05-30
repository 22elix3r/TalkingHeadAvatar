import os
import sys
from pathlib import Path
import re

# Helps reduce allocator fragmentation on long runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VHAP_ROOT = PROJECT_ROOT / "VHAP"

# Add VHAP to path
sys.path.append(str(VHAP_ROOT))

from vhap.model.tracker import GlobalTracker


def _parse_checkpoint_epoch(path: Path) -> int:
    m = re.search(r"tracked_flame_params_(\d+)\.npz$", path.name)
    if not m:
        return 0
    return int(m.group(1))


def _resolve_existing_path(path: Path) -> Path | None:
    if path.exists():
        return path
    candidate = (PROJECT_ROOT / path).resolve()
    if candidate.exists():
        return candidate
    return None


def _resolve_effective_checkpoint_epoch(tracked_path: Path, visited: set[Path] | None = None) -> int:
    tracked_path = tracked_path.resolve()
    local_epoch = _parse_checkpoint_epoch(tracked_path)
    if visited is None:
        visited = set()
    if tracked_path in visited:
        return local_epoch
    visited.add(tracked_path)

    config_path = tracked_path.parent / "config.yml"
    if not config_path.exists():
        return local_epoch

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.unsafe_load(f)

    parent_path = getattr(getattr(cfg, "model", None), "flame_params_path", None)
    if parent_path is None:
        return local_epoch

    parent_path = _resolve_existing_path(Path(parent_path))
    if parent_path is None:
        return local_epoch

    if not re.search(r"tracked_flame_params_\d+\.npz$", parent_path.name):
        return local_epoch

    if parent_path.resolve() == tracked_path:
        return local_epoch

    parent_epoch = _resolve_effective_checkpoint_epoch(parent_path, visited)
    return parent_epoch + local_epoch


def _find_latest_saved_checkpoint(vhap_track_root: Path) -> tuple[Path, int]:
    candidates = sorted(
        vhap_track_root.glob("*/tracked_flame_params_*.npz"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No tracked_flame_params_*.npz found under {vhap_track_root}"
        )
    latest = candidates[-1]
    return latest, _parse_checkpoint_epoch(latest)


def resume():
    vhap_track_root = PROJECT_ROOT / "output/sam_altman/vhap_track"
    tracked_path, local_checkpoint_epoch = _find_latest_saved_checkpoint(vhap_track_root)
    checkpoint_epoch = _resolve_effective_checkpoint_epoch(tracked_path)
    config_path = tracked_path.parent / "config.yml"
    target_total_epochs = 30

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.unsafe_load(f)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Aborting to avoid CPU/system-RAM fallback.")

    # Resume global refinement only.
    cfg.begin_timestep = 8556
    cfg.model.flame_params_path = tracked_path
    cfg.batch_size = 4
    cfg.async_func = False
    cfg.device = "cuda"

    # Reduce memory pressure from media logging threads/outputs.
    cfg.log.interval_media = 1000000
    cfg.log.max_num_views = 1

    # Continue only remaining epochs from the latest saved checkpoint.
    stage_name = "rgb_global_tracking" if cfg.exp.photometric else "lmk_global_tracking"
    remaining_epochs = max(1, target_total_epochs - checkpoint_epoch)
    getattr(cfg.pipeline, stage_name).num_epochs = remaining_epochs

    # Ensure FLAME relative asset paths resolve against VHAP root.
    os.chdir(VHAP_ROOT)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Resuming global refinement from: {tracked_path}")
    print(f"begin_timestep={cfg.begin_timestep}, batch_size={cfg.batch_size}, async_func={cfg.async_func}")
    print(f"device={cfg.device}, interval_media={cfg.log.interval_media}, max_num_views={cfg.log.max_num_views}")
    print(f"local_checkpoint_epoch={local_checkpoint_epoch}")
    print(f"checkpoint_epoch={checkpoint_epoch}, remaining_global_epochs={remaining_epochs}")
    print(f"Working directory: {Path.cwd()}")
    print(f"CUDA GPU: {gpu_name} ({gpu_mem_gb:.2f} GiB)")

    tracker = GlobalTracker(cfg)
    tracker.global_step = 0
    tracker.logger.info(
        "Skipping sequential/eval pass; resuming global refinement from latest saved checkpoint."
    )

    dataloader = DataLoader(
        tracker.dataset,
        batch_size=cfg.batch_size if not tracker.dataset.batchify_all_views else None,
        shuffle=True,
        num_workers=2,
    )
    tracker.optimize_stage(stage=stage_name, dataloader=dataloader, lr_scale=0.1)
    tracker.logger.info("Global refinement stage completed.")


if __name__ == "__main__":
    resume()
