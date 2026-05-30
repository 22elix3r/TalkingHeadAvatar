import numpy as np

from runtime_animation import RuntimeAnimationConfig, RuntimeAnimationController


def test_runtime_animation_outputs_expected_shapes():
    basis = np.eye(4, 8, dtype=np.float32)
    controller = RuntimeAnimationController(
        n_expr=8,
        config=RuntimeAnimationConfig(fps=30, seed=1),
        expression_basis=basis,
    )

    frame = controller.step(
        expression=np.zeros(8, dtype=np.float32),
        jaw=np.array([0.1, 0.0, 0.0], dtype=np.float32),
        has_motion=True,
        emotion_mode="engaged",
        now=1.0,
    )

    assert frame.expression.shape == (8,)
    assert frame.jaw.shape == (3,)
    assert frame.rotation_delta.shape == (3,)
    assert frame.neck_delta.shape == (3,)
    assert frame.eyes_delta.shape == (6,)
    assert frame.translation_delta.shape == (3,)


def test_runtime_animation_adds_idle_pose_motion():
    controller = RuntimeAnimationController(
        n_expr=8,
        config=RuntimeAnimationConfig(fps=30, seed=1),
        expression_basis=np.eye(2, 8, dtype=np.float32),
    )

    first = controller.step(
        expression=np.zeros(8, dtype=np.float32),
        jaw=np.zeros(3, dtype=np.float32),
        has_motion=False,
        now=1.0,
    )
    second = controller.step(
        expression=np.zeros(8, dtype=np.float32),
        jaw=np.zeros(3, dtype=np.float32),
        has_motion=False,
        now=1.1,
    )

    assert np.linalg.norm(first.rotation_delta) > 0
    assert not np.allclose(first.rotation_delta, second.rotation_delta)
    assert np.linalg.norm(second.expression) > 0
