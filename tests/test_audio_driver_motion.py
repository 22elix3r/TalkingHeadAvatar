import numpy as np
import torch

from audio_driver.motion_frame import FlameMotionFrame
from audio_driver.motion_translator import MotionTranslator, MotionTranslatorLoss


def test_motion_translator_can_predict_full_flame_channels():
    model = MotionTranslator(
        audio_dim=16,
        n_expr=8,
        n_layers=1,
        n_heads=4,
        ff_dim=32,
        output_channels=("expr", "jaw", "rotation", "neck", "eyes", "translation"),
        causal=True,
    )

    pred = model.forward_dict(torch.zeros(2, 5, 16))

    assert pred["expr"].shape == (2, 5, 8)
    assert pred["jaw"].shape == (2, 5, 3)
    assert pred["rotation"].shape == (2, 5, 3)
    assert pred["neck"].shape == (2, 5, 3)
    assert pred["eyes"].shape == (2, 5, 6)
    assert pred["translation"].shape == (2, 5, 3)


def test_motion_translator_loss_accepts_named_channels():
    criterion = MotionTranslatorLoss()
    pred = {
        "expr": torch.zeros(1, 4, 8),
        "jaw": torch.zeros(1, 4, 3),
        "rotation": torch.zeros(1, 4, 3),
    }
    target = {
        "expr": torch.ones(1, 4, 8),
        "jaw": torch.ones(1, 4, 3),
        "rotation": torch.ones(1, 4, 3),
    }

    losses = criterion(pred, target)

    assert losses["total"].item() > 0
    assert losses["expr"].item() == 1.0
    assert losses["jaw"].item() == 1.0
    assert losses["rotation"].item() == 1.0


def test_flame_motion_frame_coerces_and_scales_channels():
    frame = FlameMotionFrame.from_channels(
        {
            "expr": np.ones(4, dtype=np.float32),
            "jaw": np.ones(10, dtype=np.float32),
            "eyes": np.ones(6, dtype=np.float32),
        },
        n_expr=8,
    ).scaled(jaw_scale=2.0, eyes_scale=0.5)

    assert frame.expression.shape == (8,)
    assert frame.jaw.shape == (3,)
    assert frame.eyes.shape == (6,)
    assert np.allclose(frame.expression[:4], 1.0)
    assert np.allclose(frame.jaw, 0.45)
    assert np.allclose(frame.eyes, 0.25)
