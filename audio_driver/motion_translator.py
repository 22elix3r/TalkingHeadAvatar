"""
Motion Translator — HuBERT Features → FLAME Parameters
=======================================================
Speaker-specific Transformer that maps audio features to FLAME expression
coefficients and jaw pose (axis-angle).

Architecture:
  - 4-layer Transformer Encoder (causal mask optional for streaming)
  - Expression head: linear → (T, n_expr)
  - Jaw pose head:   linear → (T, 3)

Training: supervised regression on VHAP-tracked FLAME params paired
          with corresponding audio windows.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path


class MotionTranslator(nn.Module):
    """
    Maps audio features → FLAME expression + jaw parameters.

    Args:
        audio_dim:      Dimensionality of input audio features (1024 for HuBERT-Large).
        n_expr:         Number of FLAME expression coefficients (default 100).
        n_jaw:          Number of jaw pose dims (3 for axis-angle).
        n_layers:       Transformer encoder depth.
        n_heads:        Multi-head attention heads.
        ff_dim:         Feed-forward hidden dim.
        dropout:        Dropout rate.
        causal:         If True, use causal attention mask (required for streaming).
    """

    def __init__(
        self,
        audio_dim: int = 1024,
        n_expr: int = 100,
        n_jaw: int = 3,
        n_layers: int = 4,
        n_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.n_expr = n_expr
        self.n_jaw = n_jaw
        self.causal = causal

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=audio_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.expr_head = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, n_expr),
        )
        self.jaw_head = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, n_jaw),
            nn.Tanh(),  # jaw angle bounded to (-1, 1) rad ≈ ±57°
        )

        self._init_weights()

    def _init_weights(self):
        """Small-magnitude init for the output heads to prevent training instability."""
        for head in [self.expr_head, self.jaw_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _make_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask (True = ignore) for causal attention."""
        mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(
        self,
        audio_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_features: (B, T, audio_dim)

        Returns:
            expression: (B, T, n_expr) — FLAME expression coefficients
            jaw_pose:   (B, T, n_jaw)  — jaw rotation (axis-angle, radians)
        """
        B, T, _ = audio_features.shape

        mask = None
        if self.causal:
            mask = self._make_causal_mask(T, audio_features.device)

        encoded = self.temporal_encoder(audio_features, mask=mask)  # (B, T, audio_dim)
        expression = self.expr_head(encoded)                          # (B, T, n_expr)
        jaw_pose = self.jaw_head(encoded)                             # (B, T, n_jaw)
        return expression, jaw_pose

    # ──────────────────────────────────────────────
    # Persistence helpers
    # ──────────────────────────────────────────────

    def save(self, path: str | Path):
        """Save model weights + config to a single file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": {
                "audio_dim": self.audio_dim,
                "n_expr": self.n_expr,
                "n_jaw": self.n_jaw,
                "causal": self.causal,
            },
            "state_dict": self.state_dict(),
        }, str(path))
        print(f"[MotionTranslator] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cuda") -> "MotionTranslator":
        """Load a checkpoint saved with .save()."""
        path = Path(path)
        ckpt = torch.load(str(path), map_location=device, weights_only=True)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model.to(device)


# ──────────────────────────────────────────────────────────
# Training Loss
# ──────────────────────────────────────────────────────────

class MotionTranslatorLoss(nn.Module):
    """
    Combined loss for Motion Translator training:
      L = λ_expr * ||expr_pred - expr_gt||₁
        + λ_jaw  * ||jaw_pred  - jaw_gt||₁
        + λ_vel  * ||Δexpr_pred - Δexpr_gt||₁   (velocity smoothing)

    All terms use L1 (MAE) which is more robust than MSE for noisy FLAME tracks.
    """

    def __init__(
        self,
        lambda_expr: float = 1.0,
        lambda_jaw: float = 5.0,   # jaw is low-dim, upweight it
        lambda_vel: float = 0.1,
    ):
        super().__init__()
        self.lambda_expr = lambda_expr
        self.lambda_jaw = lambda_jaw
        self.lambda_vel = lambda_vel

    def forward(
        self,
        expr_pred: torch.Tensor,
        jaw_pred: torch.Tensor,
        expr_gt: torch.Tensor,
        jaw_gt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            expr_pred, expr_gt: (B, T, n_expr)
            jaw_pred,  jaw_gt:  (B, T, 3)

        Returns:
            dict with 'total' and individual term losses.
        """
        l_expr = torch.abs(expr_pred - expr_gt).mean()
        l_jaw = torch.abs(jaw_pred - jaw_gt).mean()

        # Velocity loss on expressions (temporal smoothness)
        if expr_pred.shape[1] > 1:
            vel_pred = expr_pred[:, 1:] - expr_pred[:, :-1]
            vel_gt = expr_gt[:, 1:] - expr_gt[:, :-1]
            l_vel = torch.abs(vel_pred - vel_gt).mean()
        else:
            l_vel = torch.zeros(1, device=expr_pred.device)

        total = (
            self.lambda_expr * l_expr
            + self.lambda_jaw * l_jaw
            + self.lambda_vel * l_vel
        )

        return {
            "total": total,
            "expr": l_expr,
            "jaw": l_jaw,
            "vel": l_vel,
        }
