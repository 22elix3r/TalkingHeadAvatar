import sys
import os
import json
import math
from pathlib import Path
import torch
import numpy as np
from dataclasses import dataclass

# Add GaussianAvatars to path
PROJECT_ROOT = Path(__file__).resolve().parent
GA_PATH = PROJECT_ROOT / "GaussianAvatars"
if str(GA_PATH) not in sys.path:
    sys.path.insert(0, str(GA_PATH))

from scene import FlameGaussianModel
from gaussian_renderer import render
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov


@dataclass
class _PipelineParams:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False


class AvatarRenderer:
    """
    Real-time 3DGS Avatar Renderer using FlameGaussianModel.

    Handles topology mismatches between GeoAvatar-trained checkpoints
    (5273 verts / 10360 faces) and our FlameHead+teeth model (5143 / 10144)
    transparently via FlameGaussianModel.load_ply() reconciliation logic.
    """

    def __init__(self, ckpt_path: str, resolution: int = 512, device: str = "cuda"):
        self.ckpt_path = Path(ckpt_path).expanduser().resolve()
        self.resolution = resolution
        self.device = torch.device(device)

        # ── 1. Discover PLY file ─────────────────────────────────────────
        ply_path = self._find_ply(self.ckpt_path)
        print(f"[AvatarRenderer] PLY: {ply_path}")

        # ── 2. Parse sh_degree from cfg_args ────────────────────────────
        sh_degree = self._parse_sh_degree(self.ckpt_path)
        print(f"[AvatarRenderer] sh_degree={sh_degree}")

        # ── 3. Build FlameGaussianModel ──────────────────────────────────
        print("[AvatarRenderer] Loading FlameGaussianModel…")
        self.gaussians = FlameGaussianModel(sh_degree=sh_degree)
        self.gaussians.load_ply(str(ply_path), has_target=False)

        # Cache vertex count for render() param sizing
        self.n_expr = self.gaussians.flame_param["expr"].shape[-1]
        print(f"[AvatarRenderer] n_expr={self.n_expr}, "
              f"n_timesteps={self.gaussians.num_timesteps}")

        # ── 4. Use a real trained camera/base timestep when available ───
        self.camera_entry = self._load_camera_entry(self.ckpt_path)
        self.base_timestep = self._camera_timestep(self.camera_entry)
        self.base_timestep %= max(int(self.gaussians.num_timesteps), 1)
        print(f"[AvatarRenderer] base_timestep={self.base_timestep}")

        # Initialise mesh from the tracked base timestep. The live audio driver
        # later replaces only expression/jaw while preserving pose/translation.
        self.gaussians.select_mesh_by_timestep(self.base_timestep)

        # ── 5. Camera ───────────────────────────────────────────────────
        self.cam = self._build_camera()

        # ── 6. Pipeline params ──────────────────────────────────────────
        self.pipe = _PipelineParams()
        self.bg_color = torch.tensor([1.0, 1.0, 1.0],
                                     dtype=torch.float32, device=self.device)

        print("[AvatarRenderer] Ready.")

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _find_ply(ckpt_path: Path) -> Path:
        """Locate the highest-iteration point_cloud.ply under ckpt_path."""
        pc_dir = ckpt_path / "point_cloud"
        if not pc_dir.exists():
            raise FileNotFoundError(
                f"'point_cloud/' directory not found under {ckpt_path}")

        iters = sorted(
            int(d.name.split("_")[1])
            for d in pc_dir.iterdir()
            if d.is_dir() and d.name.startswith("iteration_")
        )
        if not iters:
            raise FileNotFoundError(
                f"No iteration_* subdirectories found in {pc_dir}")

        ply = pc_dir / f"iteration_{iters[-1]}" / "point_cloud.ply"
        if not ply.exists():
            raise FileNotFoundError(f"point_cloud.ply not found at {ply}")
        return ply

    @staticmethod
    def _parse_sh_degree(ckpt_path: Path) -> int:
        cfg = ckpt_path / "cfg_args"
        if cfg.exists():
            try:
                text = cfg.read_text()
                if "sh_degree=" in text:
                    val = text.split("sh_degree=")[1].split(",")[0].split(")")[0].strip()
                    return int(val)
            except Exception as exc:
                print(f"[AvatarRenderer] Warning: could not parse cfg_args: {exc}")
        return 0

    def _build_camera(self) -> MiniCam:
        """Build the live camera.

        Prefer the saved GaussianAvatars camera metadata because it matches the
        trained coordinate system and preserves facial detail. Fall back to the
        old fixed portrait camera only for external checkpoints without
        cameras.json.
        """
        if self.camera_entry is not None:
            return self._build_camera_from_entry(self.camera_entry)

        print("[AvatarRenderer] Warning: cameras.json missing; using fallback camera.")
        w = h = self.resolution

        # Calibrated portrait framing for GeoAvatar / GaussianAvatars checkpoints
        # Face bbox:  X[-0.09..0.12], Y[-0.18..0.14], Z[-0.32..-0.09]
        # Face center ≈ [0.006, 0.007, -0.157], height ≈ 0.32 m
        # Camera sits on +Z axis, looking toward -Z.  With FOV=30° and
        # cam-to-face distance ≈ 1.0 m the head fills ~60% of the frame.
        fovy_deg = 30.0
        fovx_deg = 30.0

        # The 3DGS rasterizer uses OpenCV convention: Y points DOWN in camera space.
        # FLAME world-space Y points UP. Applying R = diag([1,-1,1]) maps world-Y (up)
        # to camera-Y (down), producing an upright rendered portrait.
        R = np.diag([1.0, -1.0, 1.0]).astype(np.float64)

        # Camera centre: aligned to face centre [0.006, 0.007, -0.157], offset +1.0 m along +Z
        T = np.array([0.006, 0.007, 0.843], dtype=np.float64)  # cam-to-face ≈ 1.0 m


        w2v_raw  = getWorld2View2(R, T)         # returns np.ndarray (4,4)
        proj_raw = getProjectionMatrix(
            znear=0.01, zfar=100.0,
            fovX=float(np.radians(fovx_deg)),
            fovY=float(np.radians(fovy_deg)),
        )                                        # may return Tensor or ndarray

        # GS rasterizer expects *transposed* matrices
        def _to_tensor(m):
            if isinstance(m, torch.Tensor):
                return m.float().to(self.device)
            return torch.from_numpy(np.array(m, dtype=np.float32)).to(self.device)

        w2v  = _to_tensor(w2v_raw).T
        proj = _to_tensor(proj_raw).T
        full_proj = w2v.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)

        return MiniCam(
            width=w,
            height=h,
            fovy=float(np.radians(fovy_deg)),
            fovx=float(np.radians(fovx_deg)),
            znear=0.01,
            zfar=100.0,
            world_view_transform=w2v,
            full_proj_transform=full_proj,
            timestep=0,
        )

    @staticmethod
    def _load_camera_entry(ckpt_path: Path) -> dict | None:
        cameras_json = ckpt_path / "cameras.json"
        if not cameras_json.exists():
            return None
        try:
            cameras = json.loads(cameras_json.read_text())
        except Exception as exc:
            print(f"[AvatarRenderer] Warning: could not read cameras.json: {exc}")
            return None
        if not cameras:
            return None
        return cameras[0]

    @staticmethod
    def _camera_timestep(entry: dict | None) -> int:
        if entry is None:
            return 0
        image_name = str(entry.get("img_name", ""))
        frame_id = image_name.split("_", maxsplit=1)[0]
        try:
            return int(frame_id)
        except ValueError:
            return int(entry.get("id", 0))

    def _build_camera_from_entry(self, entry: dict) -> MiniCam:
        w = h = self.resolution

        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = np.array(entry["rotation"], dtype=np.float64)
        c2w[:3, 3] = np.array(entry["position"], dtype=np.float64)
        rt = np.linalg.inv(c2w)

        # camera_to_JSON stores C2W. getWorld2View2 expects R/T such that
        # world_view[:3, :3] == R.T and world_view[:3, 3] == T.
        R = rt[:3, :3].T
        T = rt[:3, 3]

        source_w = int(entry["width"])
        source_h = int(entry["height"])
        fovx = focal2fov(float(entry["fx"]), source_w)
        fovy = focal2fov(float(entry["fy"]), source_h)

        w2v_raw = getWorld2View2(R, T)
        proj_raw = getProjectionMatrix(
            znear=0.01,
            zfar=100.0,
            fovX=float(fovx),
            fovY=float(fovy),
        )

        def _to_tensor(m):
            if isinstance(m, torch.Tensor):
                return m.float().to(self.device)
            return torch.from_numpy(np.array(m, dtype=np.float32)).to(self.device)

        w2v = _to_tensor(w2v_raw).T
        proj = _to_tensor(proj_raw).T
        full_proj = w2v.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)

        return MiniCam(
            width=w,
            height=h,
            fovy=float(fovy),
            fovx=float(fovx),
            znear=0.01,
            zfar=100.0,
            world_view_transform=w2v,
            full_proj_transform=full_proj,
            timestep=self.base_timestep,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def render(self, expression: np.ndarray, jaw_pose: np.ndarray) -> np.ndarray:
        """
        Render one frame driven by FLAME expression + jaw parameters.

        Args:
            expression: (n_expr,) float32 — FLAME expression coefficients
            jaw_pose:   (3,)     float32 — jaw rotation axis-angle

        Returns:
            (H, W, 3) uint8 RGB image
        """
        flame_param = {
            "expr":        torch.from_numpy(expression).float().to(self.device).unsqueeze(0),
            "rotation":    self.gaussians.flame_param["rotation"][[self.base_timestep]],
            "neck":        self.gaussians.flame_param["neck_pose"][[self.base_timestep]],
            "jaw":         torch.from_numpy(jaw_pose).float().to(self.device).unsqueeze(0),
            "eyes":        self.gaussians.flame_param["eyes_pose"][[self.base_timestep]],
            "translation": self.gaussians.flame_param["translation"][[self.base_timestep]],
        }
        self._update_mesh_from_live_params(flame_param)

        out = render(self.cam, self.gaussians, self.pipe, self.bg_color)
        img = out["render"]                        # (3, H, W) float [0,1]
        return (img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    def _update_mesh_from_live_params(self, flame_param: dict[str, torch.Tensor]) -> None:
        stored = self.gaussians.flame_param
        timestep = self.base_timestep
        self.gaussians.timestep = timestep

        static_offset = self.gaussians._reconcile_offset(stored["static_offset"])
        if static_offset.dim() == 2:
            static_offset = static_offset.unsqueeze(0)

        dynamic_offset = self.gaussians._reconcile_offset(stored["dynamic_offset"][[timestep]])
        if dynamic_offset.dim() == 2:
            dynamic_offset = dynamic_offset.unsqueeze(0)

        verts, verts_cano = self.gaussians.flame_model(
            stored["shape"][None, ...],
            flame_param["expr"].cuda(),
            flame_param["rotation"].cuda(),
            flame_param["neck"].cuda(),
            flame_param["jaw"].cuda(),
            flame_param["eyes"].cuda(),
            flame_param["translation"].cuda(),
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=static_offset,
            dynamic_offset=dynamic_offset,
        )
        self.gaussians.update_mesh_properties(verts, verts_cano)
        if self.gaussians.enable_partwise_deformation:
            self.gaussians.apply_partwise_deformation()
        else:
            self.gaussians._deformed_world_xyz = None

    @torch.no_grad()
    def render_idle(self) -> np.ndarray:
        """Render the avatar in neutral pose."""
        expr = np.zeros(self.n_expr, dtype=np.float32)
        jaw  = np.zeros(3,          dtype=np.float32)
        return self.render(expr, jaw)

    @torch.no_grad()
    def render_timestep(self, timestep: int) -> np.ndarray:
        """Render directly from a stored FLAME timestep (no audio driver needed)."""
        t = int(timestep) % self.gaussians.num_timesteps
        self.gaussians.select_mesh_by_timestep(t)
        out = render(self.cam, self.gaussians, self.pipe, self.bg_color)
        img = out["render"]
        return (img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
