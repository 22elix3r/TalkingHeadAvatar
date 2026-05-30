# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from pathlib import Path
import numpy as np
import torch
# from vht.model.flame import FlameHead
from flame_model.flame import FlameHead

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
# from pytorch3d.transforms import matrix_to_quaternion
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
from utils.geoavatar_utils import (
    compute_aps_mask,
    assign_mouth_groups_vectorized,
    get_teeth_face_ids,
    rigging_regularization_loss,
    GROUP_NONE,
    GROUP_UPPER_TEETH,
    GROUP_LOWER_TEETH,
    LAMBDA_RIGID_DEFAULT,
    LAMBDA_FLEX_DEFAULT,
)


class FlameGaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, disable_flame_static_offset=False, not_finetune_flame_params=False, n_shape=300, n_expr=100):
        super().__init__(sh_degree)

        self.disable_flame_static_offset = disable_flame_static_offset
        self.not_finetune_flame_params = not_finetune_flame_params
        self.n_shape = n_shape
        self.n_expr = n_expr

        self.flame_model = FlameHead(
            n_shape, 
            n_expr,
            add_teeth=True,
        ).cuda()
        self.flame_param = None
        self.flame_param_orig = None

        # binding is initialized once the mesh topology is known
        if self.binding is None:
            self.binding = torch.arange(len(self.flame_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.flame_model.faces), dtype=torch.int32).cuda()

        # GeoAvatar: APS classification (set by adaptive_preallocate())
        self.rigid_mask: torch.Tensor | None = None
        self.flexible_mask: torch.Tensor | None = None
        # GeoAvatar: per-Gaussian mouth anatomy group (GROUP_NONE/UPPER/LOWER)
        self.mouth_group: torch.Tensor | None = None
        # GeoAvatar: cached teeth face IDs for group assignment
        self._teeth_upper_face_ids, self._teeth_lower_face_ids = get_teeth_face_ids(self.flame_model)

        # GeoAvatar: regularization weights (overridden by training args when --enable_aps is set)
        self.lambda_rigid = LAMBDA_RIGID_DEFAULT
        self.lambda_flex = LAMBDA_FLEX_DEFAULT
        self.enable_partwise_deformation = False
        self.enable_mouth_structure = False
        self.mouth_structure = None
        self._deformed_world_xyz: torch.Tensor | None = None
        self._group_reference_xyz: dict[int, torch.Tensor] = {}
        self._group_reference_center: dict[int, torch.Tensor] = {}
        self._group_reference_size: int = 0

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        if self.flame_param is None:
            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
            
            self.num_timesteps = max(pose_meshes) + 1  # required by viewers
            num_verts = self.flame_model.v_template.shape[0]

            if not self.disable_flame_static_offset:
                static_offset = torch.from_numpy(meshes[0]['static_offset'])
                if static_offset.shape[0] != num_verts:
                    static_offset = torch.nn.functional.pad(static_offset, (0, 0, 0, num_verts - meshes[0]['static_offset'].shape[1]))
            else:
                static_offset = torch.zeros([num_verts, 3])

            T = self.num_timesteps

            self.flame_param = {
                'shape': torch.from_numpy(meshes[0]['shape']),
                'expr': torch.zeros([T, meshes[0]['expr'].shape[1]]),
                'rotation': torch.zeros([T, 3]),
                'neck_pose': torch.zeros([T, 3]),
                'jaw_pose': torch.zeros([T, 3]),
                'eyes_pose': torch.zeros([T, 6]),
                'translation': torch.zeros([T, 3]),
                'static_offset': static_offset,
                'dynamic_offset': torch.zeros([T, num_verts, 3]),
            }

            for i, mesh in pose_meshes.items():
                self.flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
                self.flame_param['rotation'][i] = torch.from_numpy(mesh['rotation'])
                self.flame_param['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
                self.flame_param['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
                self.flame_param['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
                self.flame_param['translation'][i] = torch.from_numpy(mesh['translation'])
                # self.flame_param['dynamic_offset'][i] = torch.from_numpy(mesh['dynamic_offset'])
            
            for k, v in self.flame_param.items():
                self.flame_param[k] = v.float().cuda()
            
            self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}
        else:
            # NOTE: not sure when this happens
            import ipdb; ipdb.set_trace()
            pass
    
    def _reconcile_offset(self, offset: torch.Tensor) -> torch.Tensor:
        """
        Ensure a vertex-offset tensor matches self.flame_model.v_template's vertex count.
        GeoAvatar checkpoints include 130 extra inner-mouth vertices (5273) that our
        FlameHead+teeth mesh does not have (5143).  We simply truncate the excess.
        Handles shapes: (V, 3), (1, V, 3), (T, V, 3).
        """
        n_verts = self.flame_model.v_template.shape[0]
        if offset.dim() == 2:          # (V, 3)
            if offset.shape[0] != n_verts:
                offset = offset[:n_verts]
        elif offset.dim() == 3:        # (T, V, 3) or (1, V, 3)
            if offset.shape[1] != n_verts:
                offset = offset[:, :n_verts, :]
        return offset

    def update_mesh_by_param_dict(self, flame_param):
        if 'shape' in flame_param:
            shape = flame_param['shape']
        else:
            shape = self.flame_param['shape']

        if 'static_offset' in flame_param:
            static_offset = flame_param['static_offset']
        else:
            static_offset = self.flame_param['static_offset']
        static_offset = self._reconcile_offset(static_offset)
        # FLAME forward expects (B, V, 3); if stored as (1, V, 3) that broadcasts fine.
        # If shape is (V, 3) we need to add a batch dim.
        if static_offset.dim() == 2:
            static_offset = static_offset.unsqueeze(0)

        verts, verts_cano = self.flame_model(
            shape[None, ...],
            flame_param['expr'].cuda(),
            flame_param['rotation'].cuda(),
            flame_param['neck'].cuda(),
            flame_param['jaw'].cuda(),
            flame_param['eyes'].cuda(),
            flame_param['translation'].cuda(),
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=static_offset,
        )
        self.update_mesh_properties(verts, verts_cano)

    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep
        flame_param = self.flame_param_orig if original and self.flame_param_orig != None else self.flame_param

        # Reconcile offset vertex dimensions (GeoAvatar ckpts have 130 extra mouth verts)
        static_offset = self._reconcile_offset(flame_param['static_offset'])
        # FLAME forward expects (B, V, 3); (1, V, 3) broadcasts correctly.
        if static_offset.dim() == 2:
            static_offset = static_offset.unsqueeze(0)

        dynamic_offset = self._reconcile_offset(flame_param['dynamic_offset'][[timestep]])
        if dynamic_offset.dim() == 2:
            dynamic_offset = dynamic_offset.unsqueeze(0)

        verts, verts_cano = self.flame_model(
            flame_param['shape'][None, ...],
            flame_param['expr'][[timestep]],
            flame_param['rotation'][[timestep]],
            flame_param['neck_pose'][[timestep]],
            flame_param['jaw_pose'][[timestep]],
            flame_param['eyes_pose'][[timestep]],
            flame_param['translation'][[timestep]],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=static_offset,
            dynamic_offset=dynamic_offset,
        )
        self.update_mesh_properties(verts, verts_cano)
        if self.enable_partwise_deformation:
            self.apply_partwise_deformation()
        else:
            self._deformed_world_xyz = None

    @property
    def get_xyz(self):
        if self._deformed_world_xyz is not None:
            return self._deformed_world_xyz
        return GaussianModel.get_xyz.fget(self)
    
    def update_mesh_properties(self, verts, verts_cano):
        faces = self.flame_model.faces
        triangles = verts[:, faces]

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True)
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

        # for mesh regularization
        self.verts_cano = verts_cano
    
    def compute_dynamic_offset_loss(self):
        # loss_dynamic = (self.flame_param['dynamic_offset'][[self.timestep]] - self.flame_param_orig['dynamic_offset'][[self.timestep]]).norm(dim=-1)
        loss_dynamic = self.flame_param['dynamic_offset'][[self.timestep]].norm(dim=-1)
        return loss_dynamic.mean()
    
    def compute_laplacian_loss(self):
        # offset = self.flame_param['static_offset'] + self.flame_param['dynamic_offset'][[self.timestep]]
        offset = self.flame_param['dynamic_offset'][[self.timestep]]
        verts_wo_offset = (self.verts_cano - offset).detach()
        verts_w_offset = verts_wo_offset + offset

        L = self.flame_model.laplacian_matrix[None, ...].detach()  # (1, V, V)
        lap_wo = L.bmm(verts_wo_offset).detach()
        lap_w = L.bmm(verts_w_offset)
        diff = (lap_wo - lap_w) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff.mean()
    
    def training_setup(self, training_args):
        super().training_setup(training_args)
        self.enable_partwise_deformation = bool(
            getattr(training_args, "enable_partwise_deformation", self.enable_partwise_deformation)
        )
        self.enable_mouth_structure = bool(
            getattr(training_args, "enable_mouth_structure", self.enable_mouth_structure)
        )

        if self.not_finetune_flame_params:
            return

        # # shape
        # self.flame_param['shape'].requires_grad = True
        # param_shape = {'params': [self.flame_param['shape']], 'lr': 1e-5, "name": "shape"}
        # self.optimizer.add_param_group(param_shape)

        # pose
        self.flame_param['rotation'].requires_grad = True
        self.flame_param['neck_pose'].requires_grad = True
        self.flame_param['jaw_pose'].requires_grad = True
        self.flame_param['eyes_pose'].requires_grad = True
        params = [
            self.flame_param['rotation'],
            self.flame_param['neck_pose'],
            self.flame_param['jaw_pose'],
            self.flame_param['eyes_pose'],
        ]
        param_pose = {'params': params, 'lr': training_args.flame_pose_lr, "name": "pose"}
        self.optimizer.add_param_group(param_pose)

        # translation
        self.flame_param['translation'].requires_grad = True
        param_trans = {'params': [self.flame_param['translation']], 'lr': training_args.flame_trans_lr, "name": "trans"}
        self.optimizer.add_param_group(param_trans)
        
        # expression
        self.flame_param['expr'].requires_grad = True
        param_expr = {'params': [self.flame_param['expr']], 'lr': training_args.flame_expr_lr, "name": "expr"}
        self.optimizer.add_param_group(param_expr)

        # # static_offset
        # self.flame_param['static_offset'].requires_grad = True
        # param_static_offset = {'params': [self.flame_param['static_offset']], 'lr': 1e-6, "name": "static_offset"}
        # self.optimizer.add_param_group(param_static_offset)

        # # dynamic_offset
        # self.flame_param['dynamic_offset'].requires_grad = True
        # param_dynamic_offset = {'params': [self.flame_param['dynamic_offset']], 'lr': 1.6e-6, "name": "dynamic_offset"}
        # self.optimizer.add_param_group(param_dynamic_offset)

    # ──────────────────────────────────────────────
    # GeoAvatar Phase 3 methods
    # ──────────────────────────────────────────────

    def adaptive_preallocate(self):
        """
        APS (Adaptive Pre-allocation Stage): classify each Gaussian as
        rigid (face skin) or flexible (hair, ears, neck) based on the
        magnitude of its local displacement from its bound face centre.

        Sets self.rigid_mask and self.flexible_mask (both (N,) bool).
        Call once after the warm-up phase (~30K iterations).
        """
        if self._xyz is None or self.binding is None:
            raise RuntimeError("adaptive_preallocate() called before Gaussians are initialised")

        with torch.no_grad():
            self.rigid_mask, self.flexible_mask = compute_aps_mask(
                self._xyz.detach(), self.binding
            )

        n_rigid = self.rigid_mask.sum().item()
        n_flex = self.flexible_mask.sum().item()
        print(
            f"[APS] rigid={n_rigid} ({100*n_rigid/(n_rigid+n_flex):.1f}%)  "
            f"flexible={n_flex} ({100*n_flex/(n_rigid+n_flex):.1f}%)"
        )

    def assign_mouth_part_groups(self):
        """
        Assign each Gaussian to an anatomical mouth group:
          GROUP_NONE (0)          — general face / hair / neck
          GROUP_UPPER_TEETH (1)   — upper teeth + palate mesh faces
          GROUP_LOWER_TEETH (2)   — lower teeth + floor mesh faces

        Sets self.mouth_group (N,) int32.
        Call once after APS.
        """
        if self.binding is None:
            raise RuntimeError("assign_mouth_part_groups() called before mesh binding")

        self.mouth_group = assign_mouth_groups_vectorized(
            self.binding,
            self._teeth_upper_face_ids,
            self._teeth_lower_face_ids,
        )

        n_upper = (self.mouth_group == GROUP_UPPER_TEETH).sum().item()
        n_lower = (self.mouth_group == GROUP_LOWER_TEETH).sum().item()
        print(f"[MouthGroups] upper_teeth={n_upper}  lower_teeth={n_lower}")
        self._group_reference_xyz.clear()
        self._group_reference_center.clear()
        self._group_reference_size = 0

    def apply_partwise_deformation(self):
        """
        For Gaussians assigned to upper/lower teeth groups, override
        their world-space positions with the rigid body transform of
        the corresponding jaw part.

        Upper teeth (GROUP_UPPER_TEETH): already fixed to FLAME maxilla
        → no override needed; handled by standard binding.

        Lower teeth (GROUP_LOWER_TEETH): move with jaw; apply a uniform
        rigid offset equal to the mean displacement of all lower-teeth-
        bound Gaussians, so the group moves as a single rigid body.

        This is called inside select_mesh_by_timestep() after
        update_mesh_properties() so that face_center is fresh.
        """
        if self.mouth_group is None or self.face_center is None:
            self._deformed_world_xyz = None
            return

        base_world = GaussianModel.get_xyz.fget(self)
        deformed_world = base_world.clone()

        if self._group_reference_size != deformed_world.shape[0]:
            self._group_reference_xyz.clear()
            self._group_reference_center.clear()
            self._group_reference_size = deformed_world.shape[0]

        for group_id in (GROUP_UPPER_TEETH, GROUP_LOWER_TEETH):
            group_mask = self.mouth_group == group_id
            if not group_mask.any():
                continue

            group_binding = self.binding[group_mask].long()
            group_binding = group_binding.clamp(0, self.face_center.shape[0] - 1)
            current_center = self.face_center[group_binding].mean(dim=0)

            if group_id not in self._group_reference_xyz:
                self._group_reference_center[group_id] = current_center.detach().clone()
                self._group_reference_xyz[group_id] = deformed_world[group_mask].detach().clone()

            translation = current_center - self._group_reference_center[group_id]
            deformed_world[group_mask] = self._group_reference_xyz[group_id] + translation

        self._deformed_world_xyz = deformed_world

    def compute_rigging_loss(
        self,
        lambda_rigid: float | None = None,
        lambda_flex: float | None = None,
    ) -> torch.Tensor:
        """
        GeoAvatar rigging regularization loss (Phase 3.4).
        Returns a scalar tensor. Returns 0 if APS hasn't run yet.
        """
        if self.rigid_mask is None:
            return torch.tensor(0.0, device=self._xyz.device)

        lr = lambda_rigid if lambda_rigid is not None else self.lambda_rigid
        lf = lambda_flex  if lambda_flex  is not None else self.lambda_flex

        return rigging_regularization_loss(
            self._xyz,
            self.rigid_mask,
            self.flexible_mask,
            lambda_rigid=lr,
            lambda_flex=lf,
        )

    def save_ply(self, path):
        super().save_ply(path)

        npz_path = Path(path).parent / "flame_param.npz"
        flame_param = {k: v.detach().cpu().numpy() for k, v in self.flame_param.items()}
        np.savez(str(npz_path), **flame_param)

    def load_ply(self, path, **kwargs):
        super().load_ply(path)

        if not kwargs['has_target']:
            # When there is no target motion specified, use the finetuned FLAME parameters.
            # This operation overwrites the FLAME parameters loaded from the dataset.
            npz_path = Path(path).parent / "flame_param.npz"
            flame_param = np.load(str(npz_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items()}

            self.flame_param = flame_param
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers

            # ── Reconcile topology mismatch (GeoAvatar ckpts vs our FlameHead) ──────────
            # GeoAvatar-trained checkpoints include 130 extra inner-mouth vertices
            # (total 5273) and 216 extra faces (total 10360) that our FlameHead
            # with add_teeth=True does not produce (5143 verts, 10144 faces).
            # Strategy:
            #   1. Truncate static_offset / dynamic_offset to our vertex count.
            #   2. Clamp Gaussian binding face IDs that reference non-existent faces.
            n_verts = self.flame_model.v_template.shape[0]
            n_faces = self.flame_model.faces.shape[0]

            for key in ('static_offset', 'dynamic_offset'):
                if key in self.flame_param:
                    t = self.flame_param[key]
                    # shapes: static (1, V, 3) | dynamic (T, V, 3)
                    if t.dim() == 3 and t.shape[1] != n_verts:
                        n_excess = t.shape[1] - n_verts
                        if n_excess > 0:
                            print(f"[FlameGaussianModel] Truncating {key} "
                                  f"{tuple(t.shape)} → [:, :{n_verts}, :] "
                                  f"(dropped {n_excess} extra mouth verts from GeoAvatar ckpt)")
                        self.flame_param[key] = t[:, :n_verts, :]
                    elif t.dim() == 2 and t.shape[0] != n_verts:
                        self.flame_param[key] = t[:n_verts, :]

            # Clamp any binding face IDs that exceed our face count
            if self.binding is not None:
                oob_mask = self.binding >= n_faces
                n_oob = oob_mask.sum().item()
                if n_oob > 0:
                    print(f"[FlameGaussianModel] Clamping {n_oob} out-of-bounds Gaussian "
                          f"bindings (max face id in ckpt: {self.binding.max().item()}, "
                          f"our n_faces: {n_faces}) → clamped to n_faces-1")
                    self.binding = self.binding.clamp(max=n_faces - 1)

                # Rebuild binding_counter from scratch so it's consistent
                self.binding_counter = torch.zeros(n_faces, dtype=torch.int32, device='cuda')
                self.binding_counter.scatter_add_(
                    0, self.binding.long(),
                    torch.ones(self.binding.shape[0], dtype=torch.int32, device='cuda')
                )
        
        if 'motion_path' in kwargs and kwargs['motion_path'] is not None:
            # When there is a motion sequence specified, load only dynamic parameters.
            motion_path = Path(kwargs['motion_path'])
            flame_param = np.load(str(motion_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items() if v.dtype == np.float32}

            self.flame_param = {
                # keep the static parameters
                'shape': self.flame_param['shape'],
                'static_offset': self.flame_param['static_offset'],
                # update the dynamic parameters
                'translation': flame_param['translation'],
                'rotation': flame_param['rotation'],
                'neck_pose': flame_param['neck_pose'],
                'jaw_pose': flame_param['jaw_pose'],
                'eyes_pose': flame_param['eyes_pose'],
                'expr': flame_param['expr'],
                'dynamic_offset': flame_param['dynamic_offset'],
            }
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
        
        if 'disable_fid' in kwargs and len(kwargs['disable_fid']) > 0:
            mask = (self.binding[:, None] != kwargs['disable_fid'][None, :]).all(-1)

            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]
