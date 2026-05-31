"""
Motion Translator — HuBERT Features → FLAME Motion
==================================================
Speaker-specific Transformer that maps audio features to FLAME motion
channels.

Version-1 checkpoints predict expression + jaw only. Version-2 checkpoints can
also predict head rotation, neck pose, eye pose, and translation, while keeping
the old tuple-return API compatible.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


MOTION_CHANNEL_DIMS = {
    "expr": 100,
    "jaw": 3,
    "rotation": 3,
    "neck": 3,
    "eyes": 6,
    "translation": 3,
}


class MotionTranslator(nn.Module):
    """
    Maps audio features to ordered FLAME motion channels.

    Args:
        audio_dim: Dimensionality of HuBERT features.
        n_expr: Number of FLAME expression coefficients.
        n_jaw: Number of jaw pose dims.
        output_channels: Ordered channels to predict. Defaults to expression
            and jaw for legacy checkpoint compatibility.
        causal: If True, use a causal attention mask for streaming.
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
        output_channels: tuple[str, ...] | list[str] | None = None,
        channel_dims: dict[str, int] | None = None,
    ):
        super().__init__()
        self.audio_dim = int(audio_dim)
        self.n_expr = int(n_expr)
        self.n_jaw = int(n_jaw)
        self.causal = bool(causal)
        self.output_channels = tuple(output_channels or ("expr", "jaw"))
        self.channel_dims = dict(MOTION_CHANNEL_DIMS)
        self.channel_dims["expr"] = self.n_expr
        self.channel_dims["jaw"] = self.n_jaw
        if channel_dims:
            self.channel_dims.update({str(k): int(v) for k, v in channel_dims.items()})

        unknown = [name for name in self.output_channels if name not in self.channel_dims]
        if unknown:
            raise ValueError(f"Unknown motion output channel(s): {unknown}")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.audio_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.heads = nn.ModuleDict()
        for name in self.output_channels:
            layers: list[nn.Module] = [
                nn.LayerNorm(self.audio_dim),
                nn.Linear(self.audio_dim, self.channel_dims[name]),
            ]
            if name in {"jaw", "rotation", "neck", "eyes"}:
                layers.append(nn.Tanh())
            self.heads[name] = nn.Sequential(*layers)

        # Backward-compatible names for code that expects these attributes.
        self.expr_head = self.heads["expr"] if "expr" in self.heads else None
        self.jaw_head = self.heads["jaw"] if "jaw" in self.heads else None

        self._init_weights()

    def _init_weights(self) -> None:
        for head in self.heads.values():
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _make_causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, audio_features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Args:
            audio_features: (B, T, audio_dim)

        Returns:
            Tuple of tensors ordered by self.output_channels.
        """
        _, length, _ = audio_features.shape
        mask = self._make_causal_mask(length, audio_features.device) if self.causal else None
        encoded = self.temporal_encoder(audio_features, mask=mask)
        return tuple(self.heads[name](encoded) for name in self.output_channels)

    def forward_dict(self, audio_features: torch.Tensor) -> dict[str, torch.Tensor]:
        return dict(zip(self.output_channels, self.forward(audio_features)))

    def save(self, path: str | Path) -> None:
        """Save model weights + config to a single file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": {
                    "model_version": 2,
                    "audio_dim": self.audio_dim,
                    "n_expr": self.n_expr,
                    "n_jaw": self.n_jaw,
                    "causal": self.causal,
                    "output_channels": list(self.output_channels),
                    "channel_dims": {
                        name: self.channel_dims[name] for name in self.output_channels
                    },
                },
                "state_dict": self.state_dict(),
            },
            str(path),
        )
        print(f"[MotionTranslator] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cuda") -> "MotionTranslator":
        path = Path(path)
        ckpt = torch.load(str(path), map_location=device, weights_only=True)
        config = dict(ckpt["config"])
        config.pop("model_version", None)
        model = cls(**config)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model.to(device)


class MotionTranslatorLoss(nn.Module):
    """
    Weighted L1 loss over enabled FLAME channels plus a velocity term.

    The forward method accepts either the legacy four-tensor signature or the
    v2 dict signature: loss(pred_dict, target_dict).
    """

    def __init__(
        self,
        lambda_expr: float = 1.0,
        lambda_jaw: float = 5.0,
        lambda_vel: float = 0.1,
        channel_weights: dict[str, float] | None = None,
        velocity_channels: tuple[str, ...] | list[str] | None = None,
    ):
        super().__init__()
        self.lambda_expr = float(lambda_expr)
        self.lambda_jaw = float(lambda_jaw)
        self.lambda_vel = float(lambda_vel)
        self.channel_weights = {
            "expr": self.lambda_expr,
            "jaw": self.lambda_jaw,
            "rotation": 0.8,
            "neck": 0.6,
            "eyes": 0.5,
            "translation": 0.35,
        }
        if channel_weights:
            self.channel_weights.update({str(k): float(v) for k, v in channel_weights.items()})
        self.velocity_channels = tuple(
            velocity_channels or ("expr", "jaw", "rotation", "neck", "eyes")
        )

    def forward(
        self,
        expr_pred: torch.Tensor | dict[str, torch.Tensor],
        jaw_pred: torch.Tensor | dict[str, torch.Tensor],
        expr_gt: torch.Tensor | None = None,
        jaw_gt: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if isinstance(expr_pred, dict):
            pred = expr_pred
            assert isinstance(jaw_pred, dict)
            gt = jaw_pred
        else:
            assert expr_gt is not None and jaw_gt is not None
            pred = {"expr": expr_pred, "jaw": jaw_pred}
            gt = {"expr": expr_gt, "jaw": jaw_gt}

        first = next(iter(pred.values()))
        total = torch.zeros((), dtype=first.dtype, device=first.device)
        losses: dict[str, torch.Tensor] = {}
        velocity_losses = []

        for name, pred_tensor in pred.items():
            if name not in gt:
                continue
            l_channel = torch.abs(pred_tensor - gt[name]).mean()
            losses[name] = l_channel
            total = total + self.channel_weights.get(name, 1.0) * l_channel
            if name in self.velocity_channels and pred_tensor.shape[1] > 1:
                vel_pred = pred_tensor[:, 1:] - pred_tensor[:, :-1]
                vel_gt = gt[name][:, 1:] - gt[name][:, :-1]
                velocity_losses.append(torch.abs(vel_pred - vel_gt).mean())

        if velocity_losses:
            l_vel = torch.stack(velocity_losses).mean()
        else:
            l_vel = torch.zeros((), dtype=first.dtype, device=first.device)

        total = total + self.lambda_vel * l_vel
        losses["vel"] = l_vel
        losses["total"] = total
        return losses
