"""
GeoAvatar Utility Functions
===========================
Shared helpers for the GeoAvatar enhancements:
  - APS (Adaptive Pre-allocation Stage) classification
  - Part-wise mouth group assignment
  - Rigging regularization loss

These utilities operate on FlameGaussianModel internals and are
imported by both the model and the training loop.
"""

import torch
import torch.nn.functional as F
from typing import Tuple

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

# Mouth anatomy group IDs
GROUP_NONE = 0
GROUP_UPPER_TEETH = 1   # upper teeth + palate (moves with skull / maxilla)
GROUP_LOWER_TEETH = 2   # lower teeth + floor  (moves with jaw / mandible)

# APS threshold percentile: Gaussians whose offset magnitude is below
# this percentile of the per-face-region distribution are "rigid".
APS_RIGID_PERCENTILE = 0.50   # median split

# Regularization weights
LAMBDA_RIGID_DEFAULT = 10.0
LAMBDA_FLEX_DEFAULT = 0.1


# ──────────────────────────────────────────────────────────
# APS: Adaptive Pre-allocation Stage
# ──────────────────────────────────────────────────────────

def compute_aps_mask(
    xyz: torch.Tensor,
    binding: torch.Tensor,
    rigid_percentile: float = APS_RIGID_PERCENTILE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Classify Gaussians as rigid (close to bound face) or flexible (far from
    bound face) using an unsupervised median-split per face region.

    Args:
        xyz:    (N, 3)  — local Gaussian positions in face-local frame.
                          For GaussianAvatars these are the raw _xyz params
                          (displacement from the face-centre in face-local coords).
        binding:(N,)   — face index each Gaussian is bound to.
        rigid_percentile: Gaussians below this quantile of offset magnitude
                          within each bound face are "rigid".

    Returns:
        rigid_mask:    (N,) bool — True ⟹ rigid Gaussian
        flexible_mask: (N,) bool — True ⟹ flexible Gaussian
    """
    offset_mag = xyz.norm(dim=-1)  # (N,)
    rigid_mask = torch.zeros_like(offset_mag, dtype=torch.bool)

    # Region-aware thresholding: classify within each bound face region.
    # This mirrors APS behavior more closely than a single global threshold.
    unique_faces = torch.unique(binding)
    global_threshold = torch.quantile(offset_mag, rigid_percentile)
    min_group_size = 4

    for face_id in unique_faces:
        face_mask = binding == face_id
        count = int(face_mask.sum().item())
        if count == 0:
            continue
        if count < min_group_size:
            rigid_mask[face_mask] = offset_mag[face_mask] <= global_threshold
            continue
        face_threshold = torch.quantile(offset_mag[face_mask], rigid_percentile)
        rigid_mask[face_mask] = offset_mag[face_mask] <= face_threshold

    flexible_mask = ~rigid_mask
    return rigid_mask, flexible_mask


# ──────────────────────────────────────────────────────────
# Part-wise Mouth Group Assignment
# ──────────────────────────────────────────────────────────

def assign_mouth_groups(
    binding: torch.Tensor,
    num_gaussians: int,
    teeth_upper_face_ids: torch.Tensor,
    teeth_lower_face_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Assign each Gaussian to an anatomical mouth group.

    Args:
        binding:              (N,) int32 — face index per Gaussian
        num_gaussians:        int
        teeth_upper_face_ids: (F_u,) — face IDs that belong to upper teeth mesh
        teeth_lower_face_ids: (F_l,) — face IDs that belong to lower teeth mesh

    Returns:
        mouth_group: (N,) int — GROUP_NONE / GROUP_UPPER_TEETH / GROUP_LOWER_TEETH
    """
    mouth_group = torch.zeros(num_gaussians, dtype=torch.int32, device=binding.device)

    # Upper teeth: Gaussians bound to upper-teeth faces
    upper_set = set(teeth_upper_face_ids.tolist())
    lower_set = set(teeth_lower_face_ids.tolist())

    binding_list = binding.tolist()
    for i, fid in enumerate(binding_list):
        if fid in upper_set:
            mouth_group[i] = GROUP_UPPER_TEETH
        elif fid in lower_set:
            mouth_group[i] = GROUP_LOWER_TEETH

    return mouth_group


def assign_mouth_groups_vectorized(
    binding: torch.Tensor,
    teeth_upper_face_ids: torch.Tensor,
    teeth_lower_face_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized version of assign_mouth_groups (preferred for large N).

    Returns:
        mouth_group: (N,) int32
    """
    device = binding.device
    mouth_group = torch.zeros(binding.shape[0], dtype=torch.int32, device=device)

    # Broadcast comparison: (N,) vs (F_u,)
    upper_mask = (binding.unsqueeze(1) == teeth_upper_face_ids.to(device).unsqueeze(0)).any(dim=1)
    lower_mask = (binding.unsqueeze(1) == teeth_lower_face_ids.to(device).unsqueeze(0)).any(dim=1)

    mouth_group[upper_mask] = GROUP_UPPER_TEETH
    mouth_group[lower_mask] = GROUP_LOWER_TEETH
    return mouth_group


# ──────────────────────────────────────────────────────────
# Rigging Regularization Loss
# ──────────────────────────────────────────────────────────

def rigging_regularization_loss(
    xyz: torch.Tensor,
    rigid_mask: torch.Tensor,
    flexible_mask: torch.Tensor,
    lambda_rigid: float = LAMBDA_RIGID_DEFAULT,
    lambda_flex: float = LAMBDA_FLEX_DEFAULT,
) -> torch.Tensor:
    """
    GeoAvatar rigging regularization:
      L_rig = λ_rigid * ||offset_rigid||² + λ_flex * ||offset_flex||²

    Rigid Gaussians (face skin) are penalised heavily for large offsets
    so they stay glued to the mesh.  Flexible Gaussians (hair, ears) get
    a weaker penalty so they can represent geometry further from the surface.

    Args:
        xyz:          (N, 3) — Gaussian positions in face-local frame (_xyz).
        rigid_mask:   (N,)   — bool, True ⟹ rigid.
        flexible_mask:(N,)   — bool, True ⟹ flexible.
        lambda_rigid: weight for rigid term.
        lambda_flex:  weight for flexible term.

    Returns:
        Scalar loss tensor.
    """
    loss = torch.tensor(0.0, device=xyz.device)

    if rigid_mask.any():
        rigid_offsets = xyz[rigid_mask]
        loss = loss + lambda_rigid * (rigid_offsets ** 2).sum(dim=-1).mean()

    if flexible_mask.any():
        flex_offsets = xyz[flexible_mask]
        loss = loss + lambda_flex * (flex_offsets ** 2).sum(dim=-1).mean()

    return loss


# ──────────────────────────────────────────────────────────
# Teeth face-ID helpers
# ──────────────────────────────────────────────────────────

def get_teeth_face_ids(flame_model) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the face indices in the combined FLAME+teeth mesh that correspond
    to upper and lower teeth respectively.

    GaussianAvatars' FlameHead appends teeth faces at the end of self.faces:
      f_teeth_upper (112 faces) then f_teeth_lower (112 faces).

    The original FLAME mesh has 9976 faces, so:
      upper_teeth_faces = [9976 .. 10087]
      lower_teeth_faces = [10088 .. 10199]

    Returns:
        teeth_upper_face_ids: (112,) long
        teeth_lower_face_ids: (112,) long
    """
    n_faces_orig = 9976  # standard FLAME 2023 face count
    n_teeth_each = 112   # from add_teeth() — 56+56 triangles per jaw

    upper_ids = torch.arange(n_faces_orig, n_faces_orig + n_teeth_each, dtype=torch.long)
    lower_ids = torch.arange(n_faces_orig + n_teeth_each, n_faces_orig + 2 * n_teeth_each, dtype=torch.long)
    return upper_ids, lower_ids
