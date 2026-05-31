#!/usr/bin/env python3
"""
Train Audio Driver — Speaker-Specific MotionTranslator
=======================================================
Trains a MotionTranslator that maps HuBERT audio features to FLAME
expression + jaw parameters for a specific subject.

Input data comes from VHAP tracking: per-frame .npz FLAME parameters
paired with the corresponding audio recording.

Usage:
    source activate_env.sh
    python scripts/train_audio_driver.py \\
        --flame_params data/MY_SUBJECT/transformsVHAP/flame_param/ \\
        --audio_path   data/MY_SUBJECT/audio.wav \\
        --output_dir   audio_driver/checkpoints/MY_SUBJECT/ \\
        --epochs 100 \\
        --lr 1e-4 \\
        --batch_size 32 \\
        --context_frames 50

Requirements:
    pip install transformers soundfile librosa
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_driver.audio_encoder import AudioEncoder
from audio_driver.motion_translator import MotionTranslator, MotionTranslatorLoss


# ──────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────

class AudioFlameDataset(Dataset):
    """
    Paired (HuBERT features, FLAME params) dataset.

    Audio is pre-encoded to HuBERT features at construction time
    to avoid repeated GPU inference during training.

    Args:
        hubert_features: (T_total, 1024) pre-encoded features.
        expr_params:     (T_total, n_expr) FLAME expression per frame.
        jaw_params:      (T_total, 3)     FLAME jaw pose per frame.
        context_frames:  Number of consecutive HuBERT frames per sample.
    """

    def __init__(
        self,
        hubert_features: torch.Tensor,
        expr_params: torch.Tensor,
        jaw_params: torch.Tensor,
        context_frames: int = 50,
    ):
        assert hubert_features.shape[0] == expr_params.shape[0] == jaw_params.shape[0], \
            "Feature and label lengths must match"
        self.features = hubert_features  # (T, 1024)
        self.expr = expr_params           # (T, n_expr)
        self.jaw = jaw_params             # (T, 3)
        self.ctx = context_frames
        self.valid_starts = range(0, len(self.features) - context_frames)

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = self.valid_starts[idx]
        return (
            self.features[i : i + self.ctx],   # (ctx, 1024)
            self.expr[i : i + self.ctx],        # (ctx, n_expr)
            self.jaw[i : i + self.ctx],         # (ctx, 3)
        )


# ──────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────

def _squeeze_param(array: np.ndarray, width: int) -> np.ndarray:
    """Return a flat parameter vector, handling VHAP's common (1, D) export shape."""
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    if flat.shape[0] < width:
        out = np.zeros(width, dtype=np.float32)
        out[: flat.shape[0]] = flat
        return out
    return flat[:width]


def _load_transform_param_files(transforms_dir: Path, split_names: list[str]) -> list[Path]:
    """Load frame-ordered flame_param paths from transforms_{split}.json files."""
    param_files: list[Path] = []
    for split in split_names:
        path = transforms_dir / f"transforms_{split}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for frame in data.get("frames", []):
            rel_path = frame.get("flame_param_path")
            if rel_path is None:
                continue
            param_files.append(transforms_dir / rel_path)
    return param_files


def load_flame_params(
    flame_param_dir: Path,
    n_expr: int = 100,
    param_files: list[Path] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load per-frame FLAME params from VHAP output, returning (T, D) arrays."""
    if param_files is None:
        param_files = sorted(flame_param_dir.glob("*.npz"))
    if not param_files:
        raise FileNotFoundError(f"No FLAME .npz files found for {flame_param_dir}")

    exprs, jaws = [], []
    for f in param_files:
        d = np.load(str(f))
        exprs.append(_squeeze_param(d["expr"], n_expr))
        jaws.append(_squeeze_param(d["jaw_pose"], 3))
    return np.stack(exprs), np.stack(jaws)   # (T, n_expr), (T, 3)


def load_audio_wav(audio_path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target_sr."""
    try:
        import librosa
        audio, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
    except ImportError:
        try:
            import soundfile as sf
            audio, sr = sf.read(str(audio_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != target_sr:
                raise RuntimeError(
                    f"Audio sample rate {sr} ≠ {target_sr}. Install librosa for resampling."
                )
        except ImportError:
            raise ImportError("Install soundfile or librosa: pip install soundfile librosa")
    return audio.astype(np.float32)


def encode_audio_with_hubert(
    audio: np.ndarray,
    encoder: AudioEncoder,
    batch_seconds: float = 10.0,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Encode a full audio array with HuBERT in batch chunks (avoids OOM).

    Returns:
        features: (T', 1024) on CPU.
    """
    chunk_samples = int(batch_seconds * sample_rate)
    all_features = []
    start = 0
    while start < len(audio):
        chunk = audio[start : start + chunk_samples]
        t = torch.from_numpy(chunk).unsqueeze(0)  # (1, N)
        feat = encoder(t).squeeze(0).cpu()         # (T', 1024)
        all_features.append(feat)
        start += chunk_samples
    return torch.cat(all_features, dim=0)


def resample_sequence_to_length(sequence: torch.Tensor, n_frames: int) -> torch.Tensor:
    """
    Linearly resample a time-major sequence to a target frame count.

    Used to place VHAP/FLAME labels onto HuBERT's ~50 Hz timeline so training
    matches the streaming inference rate.
    """
    if sequence.shape[0] == n_frames:
        return sequence
    if sequence.shape[0] < 2:
        raise ValueError(f"Need at least 2 frames, got {sequence.shape[0]}")
    x = sequence.T.unsqueeze(0)  # (1, C, T)
    y = F.interpolate(x, size=n_frames, mode="linear", align_corners=True)
    return y.squeeze(0).T.contiguous()


def _map_flame_index_to_feature_index(idx: int, n_flame_frames: int, n_feature_frames: int) -> int:
    if n_flame_frames <= 1:
        return 0
    ratio = idx / float(n_flame_frames - 1)
    return int(round(ratio * (n_feature_frames - 1)))


def build_split_indices(
    transforms_dir: Path | None,
    n_flame_frames: int,
    n_feature_frames: int,
) -> tuple[list[int], list[int]]:
    """Return train/validation HuBERT-frame indices from transform JSON splits."""
    if transforms_dir is None:
        return list(range(n_feature_frames)), []

    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for split in split_indices:
        path = transforms_dir / f"transforms_{split}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for frame in data.get("frames", []):
            idx = frame.get("timestep_index")
            if idx is not None and 0 <= int(idx) < n_flame_frames:
                split_indices[split].append(int(idx))

    def _to_feature_range(flame_indices: list[int]) -> list[int]:
        if not flame_indices:
            return []
        start = _map_flame_index_to_feature_index(min(flame_indices), n_flame_frames, n_feature_frames)
        end = _map_flame_index_to_feature_index(max(flame_indices), n_flame_frames, n_feature_frames)
        return list(range(start, end + 1))

    train_idx = _to_feature_range(split_indices["train"])
    val_idx = _to_feature_range(split_indices["val"] or split_indices["test"])
    if not train_idx:
        train_idx = list(range(n_feature_frames))
    return train_idx, val_idx


# ──────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_audio_driver] Device: {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load FLAME params
    print("Loading FLAME parameters …")
    flame_dir = Path(args.flame_params)
    transforms_dir = Path(args.transforms_dir) if args.transforms_dir else None
    exprs_np, jaws_np = load_flame_params(flame_dir, n_expr=args.n_expr)
    n_frames = exprs_np.shape[0]
    print(f"  → {n_frames} frames, {exprs_np.shape[1]} expression dims")

    # 2. Load full audio
    print("Loading audio …")
    audio_np = load_audio_wav(Path(args.audio_path))
    duration = len(audio_np) / 16000.0
    flame_fps = n_frames / duration if duration > 0 else 0.0
    print(f"  → {len(audio_np)} samples ({duration:.2f}s); FLAME rate ≈ {flame_fps:.2f} fps")

    # 3. Encode audio with HuBERT
    feature_cache = Path(args.feature_cache) if args.feature_cache else output_dir / "hubert_features_raw.pt"
    if feature_cache.exists() and not args.no_feature_cache:
        print(f"Loading cached HuBERT features: {feature_cache}")
        cache = torch.load(str(feature_cache), map_location="cpu", weights_only=True)
        cache_audio_path = str(cache.get("audio_path", ""))
        cache_audio_samples = int(cache.get("audio_samples", -1))
        cache_flame_frames = int(cache.get("n_flame_frames", -1))
        cache_matches = (
            cache_audio_path == str(Path(args.audio_path))
            and cache_audio_samples == int(len(audio_np))
            and cache_flame_frames == int(n_frames)
        )
        if cache_matches:
            features = cache["features"].float()
            print(f"  → Cached HuBERT frames: {features.shape[0]}")
        else:
            print("  → Cache metadata does not match this audio/FLAME pair; recomputing.")
            features = None
    else:
        features = None

    if features is None:
        print("Encoding with HuBERT (this may take a minute) …")
        encoder = AudioEncoder(device=device, fp16=(device.type == "cuda"))
        features = encode_audio_with_hubert(
            audio_np,
            encoder,
            batch_seconds=args.hubert_batch_seconds,
        ).float()
        print(f"  → HuBERT frames: {features.shape[0]}")
        torch.save(
            {
                "features": features,
                "n_flame_frames": n_frames,
                "audio_path": str(Path(args.audio_path)),
                "audio_samples": int(len(audio_np)),
            },
            str(feature_cache),
        )
        print(f"  → Saved HuBERT feature cache: {feature_cache}")

    n_feature_frames = features.shape[0]
    exprs_pt = resample_sequence_to_length(torch.from_numpy(exprs_np).float(), n_feature_frames)
    jaws_pt = resample_sequence_to_length(torch.from_numpy(jaws_np).float(), n_feature_frames)
    print(f"  → FLAME labels upsampled to {n_feature_frames} HuBERT frames")

    # 4. Build datasets and loaders
    train_idx, val_idx = build_split_indices(transforms_dir, n_frames, n_feature_frames)
    train_features = features[train_idx]
    train_exprs = exprs_pt[train_idx]
    train_jaws = jaws_pt[train_idx]
    dataset = AudioFlameDataset(
        train_features,
        train_exprs,
        train_jaws,
        context_frames=args.context_frames,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = None
    if val_idx and len(val_idx) > args.context_frames:
        val_dataset = AudioFlameDataset(
            features[val_idx],
            exprs_pt[val_idx],
            jaws_pt[val_idx],
            context_frames=args.context_frames,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
    print(f"  → Train frames: {len(train_idx)}, windows: {len(dataset)}, batches/epoch: {len(loader)}")
    if val_loader is not None:
        print(f"  → Val frames:   {len(val_idx)}, windows: {len(val_loader.dataset)}")
    if len(loader) == 0:
        raise RuntimeError(
            "Training dataset is empty. Reduce --context_frames or --batch_size."
        )

    # 5. Build model
    model = MotionTranslator(
        audio_dim=features.shape[1],
        n_expr=args.n_expr,
        causal=True,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  → MotionTranslator: {total_params / 1e6:.2f} M parameters")

    # 6. Optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = MotionTranslatorLoss(lambda_expr=1.0, lambda_jaw=5.0, lambda_vel=0.1)

    # 7. Resume if checkpoint exists
    best_ckpt = output_dir / "best_model.pt"
    resume_ckpt = output_dir / "last_model.pt"

    start_epoch = 0
    best_loss = float("inf")
    if resume_ckpt.exists() and not args.no_resume:
        print(f"Resuming from {resume_ckpt}")
        ckpt = torch.load(str(resume_ckpt), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", best_loss)

    # 8. Training
    print(f"\n{'='*60}")
    print(f"Training MotionTranslator for {args.epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {"total": 0, "expr": 0, "jaw": 0, "vel": 0}
        t_epoch = time.time()

        for batch_feat, batch_expr, batch_jaw in loader:
            batch_feat = batch_feat.to(device, dtype=torch.float32)
            batch_expr = batch_expr.to(device, dtype=torch.float32)
            batch_jaw = batch_jaw.to(device, dtype=torch.float32)

            pred_expr, pred_jaw = model(batch_feat)
            losses = criterion(pred_expr, pred_jaw, batch_expr, batch_jaw)

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()

        scheduler.step()
        n_batches = len(loader)
        mean_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch_feat, batch_expr, batch_jaw in val_loader:
                    batch_feat = batch_feat.to(device, dtype=torch.float32)
                    batch_expr = batch_expr.to(device, dtype=torch.float32)
                    batch_jaw = batch_jaw.to(device, dtype=torch.float32)
                    pred_expr, pred_jaw = model(batch_feat)
                    losses = criterion(pred_expr, pred_jaw, batch_expr, batch_jaw)
                    val_total += losses["total"].item()
                    val_batches += 1
            if val_batches:
                val_loss = val_total / val_batches
        elapsed = time.time() - t_epoch

        line = (
            f"Epoch {epoch+1:04d}/{args.epochs:04d}  "
            f"loss={mean_losses['total']:.5f}  "
            f"expr={mean_losses['expr']:.5f}  "
            f"jaw={mean_losses['jaw']:.5f}  "
            f"vel={mean_losses['vel']:.5f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
        )
        if val_loss is not None:
            line += f"val={val_loss:.5f}  "
        line += f"t={elapsed:.1f}s"
        print(line)

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
            "config": {"audio_dim": 1024, "n_expr": args.n_expr, "causal": True},
        }, str(resume_ckpt))

        # Save best checkpoint
        score = val_loss if val_loss is not None else mean_losses["total"]
        if score < best_loss:
            best_loss = score
            model.save(best_ckpt)
            print(f"  ★ New best: {best_loss:.5f} → saved to {best_ckpt}")

    print(f"\nTraining complete. Best loss: {best_loss:.5f}")
    print(f"Best model: {best_ckpt}")

    # Save training summary
    summary = {
        "subject": str(flame_dir.parent.name),
        "n_flame_frames": n_frames,
        "n_hubert_frames": int(features.shape[0]),
        "train_frames": len(train_idx),
        "val_frames": len(val_idx),
        "n_expr": args.n_expr,
        "epochs": args.epochs,
        "best_loss": best_loss,
        "audio_path": str(Path(args.audio_path)),
        "transforms_dir": str(transforms_dir) if transforms_dir else None,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train speaker-specific MotionTranslator")
    parser.add_argument("--flame_params", required=True,
                        help="Directory containing per-frame flame_param/*.npz files")
    parser.add_argument("--audio_path", required=True,
                        help="Path to the subject's training audio (wav/mp3/flac)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save checkpoints")
    parser.add_argument("--transforms_dir", default=None,
                        help="Optional transformsVHAP directory for train/test split alignment")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_expr", type=int, default=100,
                        help="Number of FLAME expression coefficients")
    parser.add_argument("--context_frames", type=int, default=50,
                        help="Temporal context window in HuBERT frames (50 = ~1 s)")
    parser.add_argument("--hubert_batch_seconds", type=float, default=10.0,
                        help="Seconds of audio per HuBERT encoding chunk")
    parser.add_argument("--feature_cache", default=None,
                        help="Optional path for aligned HuBERT feature cache")
    parser.add_argument("--no_feature_cache", action="store_true",
                        help="Recompute HuBERT features even when cache exists")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers; keep 0 for Python 3.14/forkserver stability")
    parser.add_argument("--no_resume", action="store_true",
                        help="Start training from scratch even if a checkpoint exists")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
