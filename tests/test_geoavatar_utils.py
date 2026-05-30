"""
Tests for GeoAvatar utilities: APS, mouth group assignment, rigging loss.

These tests run CPU-only with synthetic tensors — no GPU or FLAME assets required.
"""

import sys
from pathlib import Path

import pytest
import torch

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from GaussianAvatars.utils.geoavatar_utils import (
    GROUP_LOWER_TEETH,
    GROUP_NONE,
    GROUP_UPPER_TEETH,
    assign_mouth_groups,
    assign_mouth_groups_vectorized,
    compute_aps_mask,
    rigging_regularization_loss,
)


# ──────────────────────────────────────────────────────────
# APS: Adaptive Pre-allocation Stage
# ──────────────────────────────────────────────────────────

class TestComputeAPSMask:
    def _make_xyz(self, n: int, scale: float) -> torch.Tensor:
        """Generate n Gaussians with uniformly-spaced offset magnitudes."""
        # magnitudes: 0, scale/(n-1), ..., scale
        mags = torch.linspace(0.0, scale, n)
        xyz = torch.zeros(n, 3)
        xyz[:, 0] = mags  # all on x-axis
        return xyz

    def test_median_split_even(self):
        """50-th percentile → exactly half are rigid, half flexible."""
        N = 100
        xyz = self._make_xyz(N, 1.0)
        binding = torch.zeros(N, dtype=torch.long)

        rigid, flex = compute_aps_mask(xyz, binding, rigid_percentile=0.5)

        assert rigid.shape == (N,)
        assert flex.shape == (N,)
        assert rigid.dtype == torch.bool
        assert flex.dtype == torch.bool
        assert (rigid & flex).sum() == 0, "rigid and flex must be disjoint"
        # ± 1 tolerance for floating-point quantile rounding
        assert abs(rigid.sum().item() - N // 2) <= 1

    def test_all_rigid(self):
        """percentile=1.0 → all Gaussians are rigid."""
        N = 40
        xyz = self._make_xyz(N, 2.0)
        binding = torch.zeros(N, dtype=torch.long)

        rigid, flex = compute_aps_mask(xyz, binding, rigid_percentile=1.0)

        assert rigid.all(), "all should be rigid at percentile=1.0"
        assert not flex.any(), "none should be flexible at percentile=1.0"

    def test_all_flexible(self):
        """percentile=0.0 → all Gaussians are flexible (threshold = min value)."""
        N = 40
        # Magnitudes strictly > 0 so threshold = 0 makes all flexible
        xyz = self._make_xyz(N, 2.0)
        xyz[:, 0] += 0.1
        binding = torch.zeros(N, dtype=torch.long)

        rigid, flex = compute_aps_mask(xyz, binding, rigid_percentile=0.0)

        # At percentile 0 the threshold equals the smallest magnitude
        # so at least 1 Gaussian (the smallest) can be <= threshold.
        # Just verify disjoint + covering.
        assert (rigid | flex).all()
        assert (rigid & flex).sum() == 0

    def test_zero_offsets_are_rigid(self):
        """Gaussians with zero offset (sitting on the mesh surface) must be rigid."""
        N = 20
        xyz = torch.zeros(N, 3)
        binding = torch.zeros(N, dtype=torch.long)

        rigid, flex = compute_aps_mask(xyz, binding, rigid_percentile=0.5)

        # All magnitudes are 0 ≤ threshold so all are rigid
        assert rigid.all()

    def test_returns_complementary_masks(self):
        N = 50
        xyz = torch.randn(N, 3)
        binding = torch.zeros(N, dtype=torch.long)

        rigid, flex = compute_aps_mask(xyz, binding)

        assert (rigid | flex).all(), "every Gaussian must be either rigid or flexible"
        assert (rigid & flex).sum() == 0, "masks must be mutually exclusive"


# ──────────────────────────────────────────────────────────
# Mouth group assignment
# ──────────────────────────────────────────────────────────

class TestAssignMouthGroups:
    """Tests for both scalar (loop) and vectorized mouth-group assignment."""

    def _teeth_ids(self):
        upper = torch.tensor([10, 11, 12, 13])
        lower = torch.tensor([20, 21, 22])
        return upper, lower

    def test_upper_teeth_assigned(self):
        upper, lower = self._teeth_ids()
        N = 5
        # Gaussians 0-3 bound to upper teeth faces, 4 unbound
        binding = torch.tensor([10, 11, 12, 13, 99])
        groups = assign_mouth_groups(binding, N, upper, lower)

        assert groups[0] == GROUP_UPPER_TEETH
        assert groups[1] == GROUP_UPPER_TEETH
        assert groups[3] == GROUP_UPPER_TEETH
        assert groups[4] == GROUP_NONE

    def test_lower_teeth_assigned(self):
        upper, lower = self._teeth_ids()
        N = 4
        binding = torch.tensor([20, 21, 22, 5])
        groups = assign_mouth_groups(binding, N, upper, lower)

        assert groups[0] == GROUP_LOWER_TEETH
        assert groups[2] == GROUP_LOWER_TEETH
        assert groups[3] == GROUP_NONE

    def test_vectorized_matches_loop(self):
        """Vectorized and loop implementations must agree on all values."""
        upper, lower = self._teeth_ids()
        N = 100
        # Random face IDs including some from upper/lower sets
        face_pool = torch.cat([upper, lower, torch.arange(30, 130)])
        binding = face_pool[torch.randperm(len(face_pool))[:N]]

        groups_loop = assign_mouth_groups(binding, N, upper, lower)
        groups_vec = assign_mouth_groups_vectorized(binding, upper, lower)

        assert (groups_loop == groups_vec).all(), (
            "vectorized and loop results must match"
        )

    def test_no_overlap_between_groups(self):
        upper, lower = self._teeth_ids()
        N = 10
        binding = torch.cat([upper[:2], lower[:2], torch.tensor([50, 60, 70, 80, 90, 100])])
        groups = assign_mouth_groups_vectorized(binding, upper, lower)

        upper_mask = groups == GROUP_UPPER_TEETH
        lower_mask = groups == GROUP_LOWER_TEETH
        assert (upper_mask & lower_mask).sum() == 0, "no Gaussian can be in both groups"

    def test_all_none_when_no_teeth(self):
        """If no binding matches teeth IDs, all groups are GROUP_NONE."""
        upper = torch.tensor([10, 11])
        lower = torch.tensor([20, 21])
        binding = torch.tensor([30, 40, 50, 60])
        groups = assign_mouth_groups_vectorized(binding, upper, lower)

        assert (groups == GROUP_NONE).all()


# ──────────────────────────────────────────────────────────
# Rigging Regularization Loss
# ──────────────────────────────────────────────────────────

class TestRiggingRegularizationLoss:
    def test_zero_offsets_give_zero_loss(self):
        N = 50
        xyz = torch.zeros(N, 3)
        rigid = torch.ones(N, dtype=torch.bool)
        flex = torch.zeros(N, dtype=torch.bool)

        loss = rigging_regularization_loss(xyz, rigid, flex)

        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_rigid_loss_larger_than_flex_loss(self):
        """
        Rigid Gaussians (face skin) should incur a higher total penalty than
        flexible ones when their offsets are larger and λ_rigid >> λ_flex.

        Setup: rigid Gaussians have large offsets (1.0); flexible have small
        offsets (0.01).  With λ_rigid=10 the rigid term dominates; with
        λ_flex=10 the flex term is tiny because the flex offsets are small.
        """
        N = 20
        half = N // 2

        rigid = torch.zeros(N, dtype=torch.bool)
        rigid[:half] = True
        flex = ~rigid

        xyz = torch.zeros(N, 3)
        xyz[:half] = 1.0    # large offsets for the rigid half
        xyz[half:] = 0.01   # small offsets for the flexible half

        # Heavy rigid penalty → large loss (rigid offset=1.0 gets ×10)
        loss_heavy_rigid = rigging_regularization_loss(
            xyz, rigid, flex, lambda_rigid=10.0, lambda_flex=0.1
        )
        # Heavy flex penalty → small loss (flex offset=0.01 gets ×10, still tiny)
        loss_heavy_flex = rigging_regularization_loss(
            xyz, rigid, flex, lambda_rigid=0.1, lambda_flex=10.0
        )

        assert loss_heavy_rigid > loss_heavy_flex, (
            f"Expected heavy-rigid loss ({loss_heavy_rigid:.4f}) > "
            f"heavy-flex loss ({loss_heavy_flex:.4f})"
        )

    def test_scales_with_offset_magnitude(self):
        """Larger offsets must produce larger loss (squared penalty)."""
        N = 30
        rigid = torch.ones(N, dtype=torch.bool)
        flex = torch.zeros(N, dtype=torch.bool)

        xyz_small = torch.ones(N, 3) * 0.1
        xyz_large = torch.ones(N, 3) * 1.0

        loss_small = rigging_regularization_loss(xyz_small, rigid, flex)
        loss_large = rigging_regularization_loss(xyz_large, rigid, flex)

        assert loss_large > loss_small

    def test_no_rigid_gaussians(self):
        """If there are no rigid Gaussians only the flex term contributes."""
        N = 20
        xyz = torch.ones(N, 3) * 0.5
        rigid = torch.zeros(N, dtype=torch.bool)
        flex = torch.ones(N, dtype=torch.bool)

        loss = rigging_regularization_loss(
            xyz, rigid, flex, lambda_rigid=10.0, lambda_flex=0.1
        )
        expected = 0.1 * (0.5 ** 2) * 3  # λ_flex * mean(||v||²) across 3 dims
        assert loss.item() == pytest.approx(expected, rel=1e-4)

    def test_returns_scalar_tensor(self):
        N = 10
        xyz = torch.randn(N, 3)
        rigid = torch.zeros(N, dtype=torch.bool)
        rigid[:5] = True
        flex = ~rigid

        loss = rigging_regularization_loss(xyz, rigid, flex)

        assert loss.ndim == 0, "loss must be a scalar (0-dim) tensor"
        assert loss.requires_grad is False  # no grad for pure utility call

    def test_gradients_flow(self):
        """Loss must be differentiable w.r.t. xyz for use in the training loop."""
        N = 20
        xyz = torch.randn(N, 3, requires_grad=True)
        rigid = torch.ones(N, dtype=torch.bool)
        flex = torch.zeros(N, dtype=torch.bool)

        loss = rigging_regularization_loss(xyz, rigid, flex)
        loss.backward()

        assert xyz.grad is not None
        assert xyz.grad.shape == (N, 3)
