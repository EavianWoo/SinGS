
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import trimesh
from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes

from loguru import logger

from sings.rec.utils.general import (
    get_expon_lr_func, 
    get_cosine_annealing_lr,
    strip_symmetric,
    build_scaling_rotation,
)
from sings.rec.utils.geometry.rotations import (
    axis_angle_to_rotation_6d, 
    matrix_to_quaternion, 
    matrix_to_rotation_6d, 
    quaternion_multiply,
    quaternion_to_matrix, 
    rotation_6d_to_axis_angle, 
    rotation_6d_to_matrix,
    torch_rotation_matrix_from_vectors,
)
from sings.rec.defaults.constants import SMPL_PATH, SMPLH_PATH
from sings.rec.utils.geometry_ops import subdivide_meshes, collapse_edges

from ..utils.body_model.lbs import lbs_extra
from .modules.smpl_layer import SMPL
from .modules.smplh_layer import SMPLH

from .modules.hexplane import HexPlaneField as MultiTriPlane
from .modules.decoders import AppearanceDecoder, GeometryDecoder


class SinGS():

    def __init__(
        self, 
        sh_degree: int, 
        use_rgb: bool=False,
        isotropic=False,
        fixed_opacity=True,
        init_opacity=0.5,
        init_scale_multiplier=0.5,
        thickness_factor=1.0,
        refine_level=False, 
        
        body_template='smplh',
        n_subdivision=2,
        canonical_pose_type='da_pose',
        disable_posedirs=True,
        kplanes_config=None,
        n_features=32,
    ):

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self.scaling_multiplier = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.device = 'cuda'
        self.isotropic = isotropic
        self.fixed_opacity = fixed_opacity
        self.init_opacity = init_opacity
        self.init_scale_multiplier = init_scale_multiplier
        self.thickness_factor = thickness_factor
        self.disable_posedirs = disable_posedirs
        
        self.body_template = body_template
        self.n_subdivision = n_subdivision
        self.canonical_pose_type = canonical_pose_type
        
        # networks
        self.triplane = MultiTriPlane(planeconfig=kplanes_config)
         
        self.geometry_dec_list = []
        self.appearance_dec_list = []
        
        self.num_gs_level = 1 if refine_level else 2
        for i in range(self.num_gs_level):
            if self.num_gs_level > 1 and i == self.num_gs_level - 1: # for last refinement level
                isotropic = False
                fixed_opacity = False
            
            geometry_dec = GeometryDecoder(n_features=n_features*3, isotropic=isotropic).to(self.device)
            appearance_dec = AppearanceDecoder(n_features=n_features*3, fixed_opacity=fixed_opacity).to(self.device)
        
            self.geometry_dec_list.append(geometry_dec)
            self.appearance_dec_list.append(appearance_dec)
    
        
    def init_body_template(self, betas, ):
        if betas is not None:
            self.create_betas(betas, requires_grad=False)
        self.num_betas = betas.shape[0]
        if self.body_template == 'smpl':
            logger.info(f'Using smpl template...')
            logger.info(f'Found the shape of betas [{self.num_betas}] in given params')
            logger.info(f'Set num_betas = {self.num_betas}')
            self.smpl_template = SMPL(SMPL_PATH, num_betas=self.num_betas)
        elif self.body_template == 'smplh':
            logger.info(f'Using smplh template...')
            logger.info(f'Found the shape of betas [{self.num_betas}] in given params')
            logger.info(f'Set num_betas = {self.num_betas}')
            self.smpl_template = SMPLH(SMPLH_PATH, num_betas=self.num_betas)
        
        else:
            # TODO reliable smplx
            raise NotImplementedError
        
        if self.n_subdivision > 0:
            logger.info(f"Subdividing {self.body_template} model {self.n_subdivision} times")
            self.smpl_template.subdivide_meshes(num_subdivide=self.n_subdivision, smooth=True)
        self.smpl_template = self.smpl_template.to(self.device)
        
        for attr_name, attr_value in self.smpl_template._buffers.items():
            if attr_name in ['edges', 'faces', 'vertex_id', 'vertex_label']:
                attr_name = '_' + attr_name
                setattr(self, attr_name, attr_value.clone().to(self.device))
        
        # for refinement level
        self._level_id = torch.empty(0)
        # self.num_gs_level = self.num_gs_level # gaussian level
        self.gs_level_mark = [0] # record the number of gaussians in each level
        self.gs_level_mark.append(len(self._vertex_label))

        self.init_values = {}
        self.get_canonical_verts()
    
    
    def create_body_pose(self, body_pose, requires_grad=False):
        body_pose = axis_angle_to_rotation_6d(body_pose.reshape(-1, 3)).reshape(-1, 23*6)
        self.body_pose = nn.Parameter(body_pose, requires_grad=requires_grad)
        logger.info(f"Created body pose with shape: {body_pose.shape}, requires_grad: {requires_grad}")
        
    def create_global_orient(self, global_orient, requires_grad=False):
        global_orient = axis_angle_to_rotation_6d(global_orient.reshape(-1, 3)).reshape(-1, 6)
        self.global_orient = nn.Parameter(global_orient, requires_grad=requires_grad)
        logger.info(f"Created global_orient with shape: {global_orient.shape}, requires_grad: {requires_grad}")
        
    def create_betas(self, betas, requires_grad=False):
        self.betas = nn.Parameter(betas, requires_grad=requires_grad)
        logger.info(f"Created betas with shape: {betas.shape}, requires_grad: {requires_grad}")
        
    def create_transl(self, transl, requires_grad=False):
        self.transl = nn.Parameter(transl, requires_grad=requires_grad)
        logger.info(f"Created transl with shape: {transl.shape}, requires_grad: {requires_grad}")
        
    def create_eps_offsets(self, eps_offsets, requires_grad=False):
        logger.info(f"NOT CREATED eps_offsets with shape: {eps_offsets.shape}, requires_grad: {requires_grad}")
    
    
    @property
    def get_xyz(self):
        return self._xyz
    
    
    def state_dict(self):
        save_dict = {
            'active_sh_degree': self.active_sh_degree,
            'xyz': self._xyz,
            'triplane': self.triplane.state_dict(),
            # 'appearance_dec': self.appearance_dec.state_dict(),
            # 'geometry_dec': self.geometry_dec.state_dict(),
            'scaling_multiplier': self.scaling_multiplier,
            'max_radii2D': self.max_radii2D,
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'denom': self.denom,
            'optimizer': self.optimizer.state_dict(),
            'spatial_lr_scale': self.spatial_lr_scale,
            'betas': self.betas,
            
            # just save here now for additional gaussians
            'lbs_weights': self.lbs_weights,
            'vertex_label': self._vertex_label,
            'level_id': self._level_id,
            'gs_level_mark': self.gs_level_mark
        }
        
        for i in range(self.num_gs_level):
            save_dict.update(
                {
                    f'appearance_dec_{i}': self.appearance_dec_list[i].state_dict(),
                    f'geometry_dec_{i}': self.geometry_dec_list[i].state_dict(),
                }
            )
        
        return save_dict
    
    def load_state_dict(self, state_dict, cfg=None):
        self.active_sh_degree = state_dict['active_sh_degree']
        self._xyz = state_dict['xyz']
        self.max_radii2D = state_dict['max_radii2D']
        xyz_gradient_accum = state_dict['xyz_gradient_accum']
        denom = state_dict['denom']
        opt_dict = state_dict['optimizer']
        self.spatial_lr_scale = state_dict['spatial_lr_scale']
        
        self.triplane.load_state_dict(state_dict['triplane'])
        # self.appearance_dec.load_state_dict(state_dict['appearance_dec'])
        # self.geometry_dec.load_state_dict(state_dict['geometry_dec'])
        self.scaling_multiplier = state_dict['scaling_multiplier']
        
        self.betas = state_dict['betas']
        self.lbs_weights = state_dict['lbs_weights']
        self._vertex_label = state_dict['vertex_label']
        self._level_id = state_dict['level_id']
        self.gs_level_mark = state_dict['gs_level_mark']
        
        for i in range(self.num_gs_level):
            self.geometry_dec_list[i].load_state_dict(state_dict[f'geometry_dec_{i}'])
            self.appearance_dec_list[i].load_state_dict(state_dict[f'appearance_dec_{i}'])
            print(f'load geometry and  apperance decoders {i}')
        
        if cfg is None:
            from sings.rec.defaults.config import cfg as default_cfg
            cfg = default_cfg.human.lr
        
        self.setup_optimizer(cfg)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        try:
            self.optimizer.load_state_dict(opt_dict)
        except ValueError as e:
            logger.warning(f"Optimizer load failed: {e}")
            logger.warning("Continue without a pretrained optimizer")
    
          
    def __repr__(self):
        repr_str = "Mesh Gaussians: \n"
        repr_str += "xyz: {} \n".format(self._xyz.shape)
        repr_str += "max_radii2D: {} \n".format(self.max_radii2D.shape)
        repr_str += "xyz_gradient_accum: {} \n".format(self.xyz_gradient_accum.shape)
        repr_str += "denom: {} \n".format(self.denom.shape)
        return repr_str


    def get_gs_attrs(self, opt_geo=True, opt_app=True, clip_opacity=True, clippped_opacity_min=0.2):
        
        # TODO remove multiple levels
        tri_feats = self.triplane(self.get_xyz)

        # output by level
        curr_level = len(self.gs_level_mark) - 1
        
        xyz_offsets = torch.empty(0, 3, device=self.device)
        rot6d = torch.empty(0, 6, device=self.device)
        scales_aux = torch.empty(0, 3, device=self.device)
        scales = torch.empty(0, 3, device=self.device)
        
        opacity = torch.empty(0, 1, device=self.device)
        shs = torch.empty(0, 16, 3, device=self.device)

        for i in range(curr_level):
            level_tri_feats = tri_feats[self.gs_level_mark[i]: self.gs_level_mark[i+1]]
            if opt_geo:
                geometry_out = self.geometry_dec_list[i](level_tri_feats)
            else:
                with torch.no_grad():
                    geometry_out = self.geometry_dec_list[i](level_tri_feats)
            
            if opt_app:
                appearance_out = self.appearance_dec_list[i](level_tri_feats)
            else:
                with torch.no_grad():
                    appearance_out = self.appearance_dec_list[i](level_tri_feats)

        
            xyz_offsets = torch.vstack([xyz_offsets, geometry_out['xyz_offsets']])
            if not self.isotropic:
                rot6d = torch.vstack([rot6d, geometry_out['rotations']]) ### TODO remove rot for isotropic
            else:
                rot6d = None
            scales_aux = torch.vstack([scales_aux, geometry_out['scales_aux']])
            scales = torch.vstack([scales, geometry_out['scales']])
            # scales = torch.vstack([scales, masked_scales])
            
            opacity = torch.vstack([opacity, appearance_out['opacity']])
            shs = torch.vstack([shs, appearance_out['shs']])
            

        scales[:, -1] *= self.thickness_factor        
        scales = scales * self.scaling_multiplier
        
        # if clip_opacity:
        #     opacity = torch.clamp(opacity, clippped_opacity_min)
        
        
        xyz_canon = self.get_xyz + xyz_offsets
        self.gs_xyz_canon = xyz_canon
        
        gs_attr = {
            "xyz_canon": xyz_canon,
            "xyz_offsets": xyz_offsets,
            "rot6d_canon": rot6d,
            "scales_aux": scales_aux,
            "scales": scales,
            "opacity": opacity,
            "shs": shs
        }
        
        return gs_attr
        

    def canon_forward(self):
        gs_attr = self.get_gs_attrs()
        # self.gs_xyz_canon = self.get_xyz + gs_attr['xyz_offsets']

        return {
            'xyz_offsets': gs_attr['xyz_offsets'],
            'scales_aux': gs_attr['scales_aux'],
            'scales': gs_attr['scales'],
            
            'rot6d_canon': gs_attr['rot6d_canon'],
            'shs': gs_attr['shs'],
            'opacity': gs_attr['opacity'],
        }


    def forward(
        self,
        global_orient=None, 
        body_pose=None, 
        betas=None, 
        transl=None, 
        smpl_scale=None,
        dataset_idx=-1,
        ext_tfs=None,
        opt_geo=True,
        opt_app=True,
        clip_opacity=True,
        eval_mode=False,
        gs_attrs=None,
    ):

        if gs_attrs is None:
            gs_attrs = self.get_gs_attrs(opt_geo=opt_geo, opt_app=opt_app, clip_opacity=clip_opacity)
            
        
        gs_xyz_canon = gs_attrs['xyz_canon']
        gs_xyz_offsets = gs_attrs['xyz_offsets']

        if not self.isotropic:
            gs_rot6d_canon = gs_attrs['rot6d_canon']
            gs_rotmat_canon = rotation_6d_to_matrix(gs_rot6d_canon)
            gs_rotq_canon = matrix_to_quaternion(gs_rotmat_canon)
        else:
            gs_rot6d_canon = gs_attrs['rot6d_canon']
            gs_rotmat_canon = torch.eye(3, device=self.device).unsqueeze(0).repeat(gs_xyz_canon.shape[0], 1, 1)
            gs_rotq_canon = torch.zeros(gs_xyz_canon.shape[0], 4, device=self.device)
                
        gs_scales = gs_attrs['scales']
        gs_scales_aux = gs_attrs['scales_aux']
        
        gs_opacity = gs_attrs['opacity']
        gs_shs = gs_attrs['shs']
        
        
        if hasattr(self, 'global_orient') and global_orient is None:
            global_orient = rotation_6d_to_axis_angle(
                self.global_orient[dataset_idx].reshape(-1, 6)).reshape(3)
        
        if hasattr(self, 'body_pose') and body_pose is None:
            body_pose = rotation_6d_to_axis_angle(
                self.body_pose[dataset_idx].reshape(-1, 6)).reshape(23*3)

        if hasattr(self, 'betas') and betas is None:
            betas = self.betas
            
        if hasattr(self, 'transl') and transl is None:
            transl = self.transl[dataset_idx]
        
        # canonical -> t-pose -> posed
        # remove and reapply the blendshape
        
        if self.body_template == 'smplh':
            body_pose = body_pose[:63]

        smpl_output = self.smpl_template(
            betas=betas if betas.ndim == 2 else betas.unsqueeze(0), # 
            body_pose=body_pose if body_pose.ndim == 2 else body_pose.unsqueeze(0), # 
            global_orient=global_orient if global_orient.ndim == 2 else global_orient.unsqueeze(0), # 
            disable_posedirs=False,
            return_full_pose=True,
        )
        
        A_t2pose = smpl_output.A[0]
        A_cano2pose = A_t2pose @ self.inv_A_t2cano
        xyz_deformed, _, lbs_T, _, _ = lbs_extra(
            A_cano2pose[None], gs_xyz_canon[None],
            posedirs=None, lbs_weights=self.lbs_weights, 
            pose=smpl_output.full_pose,
            disable_posedirs=self.disable_posedirs,
            pose2rot=True
        )
        
        xyz_deformed = xyz_deformed.squeeze(0)
        lbs_T = lbs_T.squeeze(0)
        
        if smpl_scale is not None:
            xyz_deformed = xyz_deformed * smpl_scale.unsqueeze(0)
            gs_scales = gs_scales * smpl_scale.unsqueeze(0)
        
        if transl is not None:
            xyz_deformed = xyz_deformed + transl.unsqueeze(0)
        
        gs_rotmat_deformed = lbs_T[:, :3, :3] @ gs_rotmat_canon
        gs_rotq_deformed = matrix_to_quaternion(gs_rotmat_deformed)
        
        if ext_tfs is not None:
            trans, rotmat, scale = ext_tfs
            xyz_deformed = (trans[..., None] + (scale[None] * (rotmat @ xyz_deformed[..., None]))).squeeze(-1)
            gs_scales = scale * gs_scales
            
            rotq = matrix_to_quaternion(rotmat)
            gs_rotq_deformed = quaternion_multiply(rotq, gs_rotq_deformed)
            gs_rotmat_deformed = quaternion_to_matrix(gs_rotq_deformed)
        
        self.normals = torch.zeros_like(gs_xyz_canon)
        self.normals[:, 2] = 1.0
        
        normals_canon = (gs_rotmat_canon @ self.normals.unsqueeze(-1)).squeeze(-1)
        normals_deformed = (gs_rotmat_deformed @ self.normals.unsqueeze(-1)).squeeze(-1)
        
        
        if not eval_mode:
            ### update anchors
            num_level_0 = (self._level_id == 0).sum()
            vertex_normals = torch.from_numpy(self.smpl_mesh.vertex_normals.copy()).float().to(self.device)     
            mean_scales = gs_scales[:num_level_0].mean(dim=-1, keepdim=True) # -> (N, 1)
            normal_offset = mean_scales * vertex_normals / 2. # scale / 2
            self.anchor_xyz = gs_xyz_canon[:num_level_0] + normal_offset
        else:
            self.anchor_xyz = None
        
        return {
            'xyz': xyz_deformed,
            'xyz_canon': gs_xyz_canon,
            'xyz_offsets': gs_xyz_offsets,
            'xyz_anchor_canon': self.anchor_xyz,

            'scales_aux': gs_scales_aux,
            'scales': gs_scales,
            'scales_canon': gs_scales,
            
            # 'rot6d_canon': gs_rot6d_canon,
            'rotq': gs_rotq_deformed,
            'rotq_canon': gs_rotq_canon,
            # 'rotmat': gs_rotmat_deformed,
            'rotmat_canon': gs_rotmat_canon,

            'shs': gs_shs,
            'opacity': gs_opacity,
            
            'normals': normals_deformed,
            'normals_canon': normals_canon,
            'active_sh_degree': self.active_sh_degree,
            
            'level_id': self._level_id,
        }
       
        
    def forward_chunk(
        self,
        gs_attrs,
        global_orient=None, 
        body_pose=None, 
        betas=None, 
        transl=None, 
        smpl_scale=None,
        ext_tfs=None,  
        chunk_size=16  
    ):
        
        gs_xyz_canon = gs_attrs['xyz_canon']
        gs_xyz_offsets = gs_attrs['xyz_offsets']
        
        if not self.isotropic:
            gs_rot6d_canon = gs_attrs['rot6d_canon']
            gs_rotmat_canon = rotation_6d_to_matrix(gs_rot6d_canon)
            gs_rotq_canon = matrix_to_quaternion(gs_rotmat_canon)
        else:
            gs_rot6d_canon = gs_attrs['rot6d_canon']
            gs_rotmat_canon = torch.eye(3, device=self.device).unsqueeze(0).repeat(gs_xyz_canon.shape[0], 1, 1)
            gs_rotq_canon = torch.zeros(gs_xyz_canon.shape[0], 4, device=self.device)
        
        gs_scales = gs_attrs['scales']        
        gs_opacity = gs_attrs['opacity']
        gs_shs = gs_attrs['shs']
        
        
        # Expand to batch
        B = chunk_size
        gs_xyz_canon = gs_xyz_canon.unsqueeze(0).expand(B, -1, -1)  # (B, N, 3)
        gs_rotmat_canon = gs_rotmat_canon.unsqueeze(0).expand(B, -1, -1, -1)
        gs_scales = gs_scales.unsqueeze(0).expand(B, -1, -1)
        gs_opacity = gs_opacity.unsqueeze(0).expand(B, -1, -1)
        gs_shs = gs_shs.unsqueeze(0).expand(B, -1, -1, -1)
        
        
        if self.body_template == 'smplh':
            body_pose = body_pose[:, :63]
        
        smpl_output = self.smpl_template(
            betas=betas.unsqueeze(0).expand(B, -1),
            body_pose=body_pose,
            global_orient=global_orient,
            disable_posedirs=False,
            return_full_pose=True,
        )
        
        
        A_t2pose = smpl_output.A
        A_cano2pose = A_t2pose @ self.inv_A_t2cano.unsqueeze(0)
        xyz_deformed, _, lbs_T, _, _ = lbs_extra(
            A_cano2pose, 
            gs_xyz_canon, 
            posedirs=None, 
            lbs_weights=self.lbs_weights,
            pose=smpl_output.full_pose,
            disable_posedirs=self.disable_posedirs,
            pose2rot=True
        )
        
        if smpl_scale is not None:
            xyz_deformed = xyz_deformed * smpl_scale.unsqueeze(-1)
            gs_scales = gs_scales * smpl_scale.unsqueeze(-1)
        
        if transl is not None:
            xyz_deformed = xyz_deformed + transl.unsqueeze(1)
        
        gs_rotmat_deformed = lbs_T[..., :3, :3] @ gs_rotmat_canon
        gs_rotq_deformed = matrix_to_quaternion(gs_rotmat_deformed)
        
        if ext_tfs is not None:
            trans, rotmat, scale = ext_tfs
            xyz_deformed = (trans[:, None, :] + (scale[:, None] * (rotmat[:, None, ...] @ xyz_deformed[..., None]).squeeze(-1)))
            gs_scales = scale[..., None] * gs_scales
            
            rotq = matrix_to_quaternion(rotmat)
            gs_rotq_deformed = quaternion_multiply(rotq[:, None, :], gs_rotq_deformed)
        
        return {
            'xyz': xyz_deformed,
            'xyz_canon': gs_xyz_canon,
            'xyz_offsets': gs_xyz_offsets,
            
            'scales': gs_scales,
            'scales_canon': gs_scales,
            
            'rotq': gs_rotq_deformed,
            'rotq_canon': gs_rotq_canon,

            'shs': gs_shs,
            'opacity': gs_opacity,
            
            'active_sh_degree': self.active_sh_degree,
        }
        

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            logger.info(f"Going from SH degree {self.active_sh_degree} to {self.active_sh_degree + 1}")
            self.active_sh_degree += 1
            
    
    @torch.no_grad()
    def get_canonical_verts(self, canonical_pose_type='da_pose'):
        from sings.rec.datasets.utils import get_predefined_pose
        predefined_pose = get_predefined_pose(self.canonical_pose_type, device=self.device)
        
        if self.body_template == 'smplh':
            predefined_pose = predefined_pose[:, :63]
        
        smpl_output = self.smpl_template(body_pose=predefined_pose, betas=self.betas[None], disable_posedirs=False)
        canonical_verts = smpl_output.vertices[0]
        self.A_t2cano = smpl_output.A[0].detach()
        self.T_t2cano = smpl_output.T[0].detach()
        self.inv_T_t2cano = torch.inverse(self.T_t2cano)
        self.inv_A_t2cano = torch.inverse(self.A_t2cano)
        self.canonical_offsets = smpl_output.shape_offsets + smpl_output.pose_offsets
        self.canonical_offsets = self.canonical_offsets[0].detach()

        self.canonical_verts = canonical_verts.detach()
        return canonical_verts.detach()
    
    
    def train(self):
        pass
    
    
    def eval(self):
        self.gs_level_mark = [0, self.max_radii2D.shape[0]]
        pass
    
    
    def init_attrs(self, lr=1e-3, init_steps=500):
        """Init values through optmization."""
        
        try:
            self.train()
        except Exception:
            pass
        
        from sings.rec.defaults.config import cfg as default_cfg
        
        self.setup_optimizer(default_cfg.human.lr)
        optim = self.optimizer

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=100, verbose=True, factor=0.5)
        fn = torch.nn.MSELoss()

        body_pose = torch.zeros((69), device=self.device).float()
        global_orient = torch.zeros((3), device=self.device).float()
        betas = torch.zeros((10), device=self.device).float()

        gt_vals = self.initialize()

        print("====== Initialize values: ======")
        for k, v in list(gt_vals.items()):
            print(f"{k:<20} | Shape: {v.shape}" )
            gt_vals[k] = v.detach().clone().to(self.device).float()
        print("================================")

        losses = []

        for i in range(init_steps):
            if hasattr(self, 'canon_forward'):
                model_out = self.canon_forward()
            else:
                model_out = self.forward(global_orient, body_pose, betas)

            
            loss_dict = {}
            for k, v in gt_vals.items():
                if k in model_out and model_out[k] is not None:
                    if model_out[k].shape != v.shape:
                        print('Unaligned shape', k, model_out[k].shape, v.shape)
                        continue
                    loss_dict['loss_' + k] = fn(model_out[k], v)

            if len(loss_dict) == 0:
                continue

            loss = sum(loss_dict.values())

            optim.zero_grad(set_to_none=True)
            loss.backward()

            loss_str = ", ".join([f"{k}: {vv.item():.7f}" for k, vv in loss_dict.items()])
            print(f"Step {i:04d}: {loss.item():.7f} ({loss_str})", end='\r')

            optim.step()
            lr_scheduler.step(loss.item())

            losses.append(loss.item())

        logger.info("Initial optimization completed.")

        return self
    
    
    def initialize(self):
        t_pose_verts = self.get_canonical_verts()
        
        self.scaling_multiplier = torch.ones((t_pose_verts.shape[0], 1), device="cuda")
        
        xyz_offsets = torch.zeros_like(t_pose_verts)
        colors = torch.ones_like(t_pose_verts) * 0.5
        
        shs = torch.zeros((colors.shape[0], 3, 16)).float().cuda()
        shs[:, :3, 0 ] = colors
        shs[:, 3:, 1:] = 0.0
        shs = shs.transpose(1, 2).contiguous()
        
        scales = torch.zeros_like(t_pose_verts)
        for v in range(t_pose_verts.shape[0]):
            selected_edges = torch.any(self._edges == v, dim=-1)
            selected_edges_len = torch.norm(
                t_pose_verts[self._edges[selected_edges][0]] - t_pose_verts[self._edges[selected_edges][1]], 
                dim=-1
            )
            selected_edges_len *= self.init_scale_multiplier

            scales[v] = torch.log(torch.max(selected_edges_len))

        # control the thickness
        scales[..., 2] *= self.thickness_factor

        scales = torch.exp(scales)
        scales_aux = torch.log(torch.exp(scales) - 1)
        
        self.smpl_mesh = trimesh.Trimesh(vertices=t_pose_verts.cpu(), faces=self.smpl_template.faces.cpu())
        vert_normals = torch.tensor(self.smpl_mesh.vertex_normals.copy()).float().to(self.device)
        
        ### to use pytorch3d mesh_edge_loss in gs trainer
        self._mesh = Meshes(verts=[t_pose_verts], faces=[self._faces])
        
        gs_normals = torch.zeros_like(vert_normals)
        gs_normals[:, 2] = 1.0
        
        norm_rotmat = torch_rotation_matrix_from_vectors(gs_normals, vert_normals)

        rotq = matrix_to_quaternion(norm_rotmat)
        rot6d = matrix_to_rotation_6d(norm_rotmat)
                
        self.normals = gs_normals
        deformed_normals = (norm_rotmat @ gs_normals.unsqueeze(-1)).squeeze(-1)
        
        ### initial opacity 0.83
        opacity = self.init_opacity * torch.ones((t_pose_verts.shape[0], 1), dtype=torch.float, device="cuda")
        
        self.lbs_weights = self.smpl_template.lbs_weights.detach().clone()

        self.n_gs = t_pose_verts.shape[0] ### initial number = vertice after subdivision
        self._xyz = nn.Parameter(t_pose_verts.requires_grad_(True)) ### initialize gaussian from T pose
        self._level_id = torch.zeros(self.n_gs)
        
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        return {
            'xyz_offsets': xyz_offsets,
            'scales': scales,
            'scales_aux': scales_aux,
            
            'rot6d_canon': rot6d,
            'shs': shs,
            'opacity': opacity,
        }
        

    def setup_optimizer(self, cfg):
        self.cfg = cfg
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.spatial_lr_scale = cfg.smpl_spatial
        
        params = [
            {'params': [self._xyz], 'lr': cfg.position_init * cfg.smpl_spatial, "name": "xyz"},
            {'params': self.triplane.parameters(), 'lr': cfg.vembed, 'name': 'v_embed'},
        ]
        
        for i in range(self.num_gs_level):
            params.extend([
                {'params': self.geometry_dec_list[i].parameters(), 'lr': cfg.geometry, 'name': f'geometry_dec_{i}'},
                {'params': self.appearance_dec_list[i].parameters(), 'lr': cfg.appearance, 'name': f'appearance_dec_{i}'},
            ])
        
        if hasattr(self, 'global_orient') and self.global_orient.requires_grad:
            params.append({'params': self.global_orient, 'lr': cfg.smpl_pose, 'name': 'global_orient'})
        
        if hasattr(self, 'body_pose') and self.body_pose.requires_grad:
            params.append({'params': self.body_pose, 'lr': cfg.smpl_pose, 'name': 'body_pose'})
            
        if hasattr(self, 'betas') and self.betas.requires_grad:
            params.append({'params': self.betas, 'lr': cfg.smpl_betas, 'name': 'betas'})
            
        if hasattr(self, 'transl') and self.betas.requires_grad:
            params.append({'params': self.transl, 'lr': cfg.smpl_trans, 'name': 'transl'})
        
        self.non_densify_params_keys = [
            'global_orient', 'body_pose', 'betas', 'transl', 'v_embed',
        ]
        
        for i in range(self.num_gs_level):
            self.non_densify_params_keys.append(f'geometry_dec_{i}')
            self.non_densify_params_keys.append(f'appearance_dec_{i}')
                
        for param in params:
            logger.info(f"Parameter: {param['name']}, lr: {param['lr']}")

        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=cfg.position_init  * cfg.smpl_spatial,
            lr_final=cfg.position_final  * cfg.smpl_spatial,
            lr_delay_mult=cfg.position_delay_mult,
            max_steps=cfg.position_max_steps,
        )

        self.feature_scheduler = get_cosine_annealing_lr(
            lr_init=cfg.vembed,
            lr_final=cfg.vembed / 5.,
            lr_delay_steps=0,
            T_max=cfg.mlp_max_steps, 
        )
        self.geometry_scheduler = get_cosine_annealing_lr(
            lr_init=cfg.geometry,
            lr_final=cfg.geometry / 5.,
            lr_delay_steps=cfg.mlp_max_steps//2,
            T_max=cfg.mlp_max_steps, 
        )
        self.appearance_scheduler = get_cosine_annealing_lr(
            lr_init=cfg.appearance,
            lr_final=cfg.appearance / 5.,
            lr_delay_steps=cfg.mlp_max_steps//2,
            T_max=cfg.mlp_max_steps,
        )
        

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
        
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == 'v_embed':
                print(f'update feature lr {lr}')
                lr = self.feature_scheduler(iteration)
                param_group['lr'] = lr
                return lr
        
        for param_group in self.optimizer.param_groups:
            if 'geometry_dec' in param_group["name"]:
                print(f'update geo lr {lr}')
                lr = self.geometry_scheduler(iteration)
                param_group['lr'] = lr
                return lr
                
        for param_group in self.optimizer.param_groups:
            if 'appearance_dec' in param_group["name"]:
                lr = self.appearance_scheduler(iteration)
                param_group['lr'] = lr
                return lr


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in self.non_densify_params_keys:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self.scaling_multiplier = self.scaling_multiplier[valid_points_mask]
        
        self.scales_tmp = self.scales_tmp[valid_points_mask]
        self.opacity_tmp = self.opacity_tmp[valid_points_mask]
        self.rotmat_tmp = self.rotmat_tmp[valid_points_mask]
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in self.non_densify_params_keys:
                continue
            
            assert len(group["params"]) == 1, f"{group['name']} has more than one param"
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_scaling_multiplier, new_opacity_tmp=None, new_scales_tmp=None, new_rotmat_tmp=None):
        d = {
            "xyz": new_xyz,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self.scaling_multiplier = torch.cat((self.scaling_multiplier, new_scaling_multiplier), dim=0)
        
        self.opacity_tmp = torch.cat([self.opacity_tmp, new_opacity_tmp], dim=0) if new_opacity_tmp is not None else None
        self.scales_tmp = torch.cat([self.scales_tmp, new_scales_tmp], dim=0) if new_scales_tmp is not None else None
        self.rotmat_tmp = torch.cat([self.rotmat_tmp, new_rotmat_tmp], dim=0) if new_rotmat_tmp is not None else None
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scale_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        scales = self.scales_tmp
        rotation = self.rotmat_tmp
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(scales, dim=1).values > scale_threshold)
        
        # only for anisotropic
        # filter elongated gaussians
        if not self.isotropic:
            med = scales.median(dim=1, keepdim=True).values 
            stdmed_mask = (((scales - med) / med).squeeze(-1) >= 1.0).any(dim=-1)
            selected_pts_mask = torch.logical_and(selected_pts_mask, stdmed_mask)
        
        stds = scales[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=torch.relu(stds))
        rots = rotation[selected_pts_mask].repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling_multiplier = self.scaling_multiplier[selected_pts_mask].repeat(N, 1) / (0.8 * N) ### 
        new_opacity_tmp = self.opacity_tmp[selected_pts_mask].repeat(N, 1)
        new_scales_tmp = self.scales_tmp[selected_pts_mask].repeat(N, 1)
        new_rotmat_tmp = self.rotmat_tmp[selected_pts_mask].repeat(N, 1, 1)
        
        self.densification_postfix(new_xyz, new_scaling_multiplier, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)


    def densify_and_clone(self, grads, grad_threshold, scale_threshold):
        # Extract points that satisfy the gradient condition
        scales = self.scales_tmp
        grad_cond = torch.norm(grads, dim=-1) >= grad_threshold
        scale_cond = torch.max(scales, dim=1).values <= scale_threshold
        
        selected_pts_mask = torch.where(grad_cond, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, scale_cond)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_scaling_multiplier = self.scaling_multiplier[selected_pts_mask]
        new_opacity_tmp = self.opacity_tmp[selected_pts_mask]
        new_scales_tmp = self.scales_tmp[selected_pts_mask]
        new_rotmat_tmp = self.rotmat_tmp[selected_pts_mask]
        
        self.densification_postfix(new_xyz, new_scaling_multiplier, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp)


    def densify_and_prune(self, human_gs_out, max_grad, min_opacity, percent_dense, densify_extent, max_screen_size, max_n_gs=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        self.opacity_tmp = human_gs_out['opacity']
        self.scales_tmp = human_gs_out['scales_canon']
        self.rotmat_tmp = human_gs_out['rotmat_canon']
        
        max_n_gs = max_n_gs if max_n_gs else self.get_xyz.shape[0] + 1
        
        if self.get_xyz.shape[0] <= max_n_gs:
            self.densify_and_clone(grads, max_grad, percent_dense * densify_extent)
            self.densify_and_split(grads, max_grad, percent_dense * densify_extent)

        prune_mask = (self.opacity_tmp < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.scales_tmp.max(dim=1).values > 0.1 * densify_extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            print('big_points_vs', big_points_vs.sum())
            print('big_points_ws', big_points_ws.sum())

        self.prune_points(prune_mask)
        self.n_gs = self.get_xyz.shape[0]
        torch.cuda.empty_cache()
    
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[:update_filter.shape[0]][update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    

    def level_up():
        pass
    
    
    def densify_and_subdivide(self, human_gs_out, grad_threshold=0.0002, scale_threshold=0.01, max_screen_size=None, max_n_gs=None, exclude=None):
        
        logger.info(f"Densification Begins")
        
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        scales = human_gs_out['scales_canon'][:, :1]
        grad_cond = torch.norm(grads, dim=-1) > grad_threshold
        scale_cond = torch.max(scales, dim=1).values > scale_threshold
        print(f"### grad selected: {grad_cond.sum()} | scale seledted: {scale_cond.sum()}")
        
        mean_scales = scales.mean()
        print(f'mean scale: {mean_scales}')
        scales_ratio = (mean_scales / scales.mean(-1)).detach()
        
        ## shs base, xyz_offsets_base
        shs = human_gs_out['shs']
        
        
        selected_pts_mask = torch.where(grad_cond, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, scale_cond)

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            # big_points_ws = self.scales_tmp.max(dim=1).values > 0.1 * extent
            selected_pts_mask = torch.logical_or(selected_pts_mask, big_points_vs)
            # prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            print(f"### super big points: {selected_pts_mask.sum().item()}")
        
        
        exclude = torch.tensor([6, 7]).long().to(self.device)
        if exclude is not None:
            non_densified = torch.isin(self._vertex_label, exclude)
            selected_pts_mask = torch.logical_and(selected_pts_mask, ~non_densified)
            print(f'Give up the densification in regions: {exclude}')
            print(f'Total Number: {len(non_densified)}')
        
        selected_pts = self._xyz[selected_pts_mask]
        selected_pts_idx = selected_pts_mask.nonzero(as_tuple=False).view(-1)

        faces = self._faces.to(self.device)
        selected_faces = torch.isin(faces, selected_pts_idx).any(dim=1)
        selected_faces_idx = selected_faces.nonzero(as_tuple=False).view(-1)
        print('selected faces', selected_faces.sum())
        
        # compute the number of gaussians to add if limit max number
        e = faces[selected_faces_idx][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        e = torch.sort(e, dim=1)[0]
        ue, counts = torch.unique(e, dim=0, return_counts=True)
        num_to_add = counts.shape[0]
        num_left = max_n_gs - self.n_gs
        
        if num_left > 0:
            if num_to_add < num_left:
                pass
            else: # according to scales
                face_scores = scales[faces[selected_faces_idx]].sum(dim=[1, 2])
                sorted_idx = torch.argsort(face_scores, descending=True)
                
                # simply select the of num_left / 3 faces here to ensure the number doesn't exceed the limit
                selected_faces_idx = selected_faces_idx[sorted_idx[:num_left // 3]]
                logger.info(f"Selected {len(selected_faces_idx)} faces to add.")
                # breakpoint()
        else: # skip 
            print('The number of gaussians is close to the maximum, just skip densification.')
        
            
        sub_vertices, sub_faces, attr = subdivide_meshes(
            vertices=human_gs_out['xyz_canon'],         
            faces=self._faces,
            face_index=selected_faces_idx,
            vertex_attributes={  
                "vertex_label": self._vertex_label, ###
                "lbs_weights": self.lbs_weights,
                "scales": torch.clip(scales.mean(-1), max=0.008),
                "shs": shs
                # 'scaling_multiplier': self.scaling_multiplier,
            }
        )
        
        self._vertex_label = attr['vertex_label'].int().detach()
        self.lbs_weights = attr['lbs_weights'].detach()
        
        self.smpl_mesh = trimesh.Trimesh(vertices=sub_vertices.cpu().numpy(), faces=sub_faces.cpu().numpy())
        unique_edges = self.smpl_mesh.edges_unique
        
        # note: use the uniue edges
        self._edges = torch.from_numpy(unique_edges.copy()).to(self.device).long()
        self._faces = sub_faces
        
        self._mesh = Meshes(verts=[sub_vertices], faces=[sub_faces])
        

        level_id = self._level_id[-1]
        num_gs_added = sub_vertices.shape[0] - scales.shape[0]
        new_level_id = torch.tensor([level_id] * num_gs_added)
        self._level_id = torch.cat([self._level_id, new_level_id])
        
        
        num_origin = scales.shape[0]
        new_xyz = sub_vertices[num_origin:]
        self.gs_level_mark[-1] = sub_vertices.shape[0]

        # rescale
        self.scaling_multiplier[selected_pts_mask] *= scales_ratio[selected_pts_mask].unsqueeze(1)
        self.scaling_multiplier = self.scaling_multiplier.detach()
        new_scaling_multiplier = torch.ones(new_xyz.shape[0], 1).to(self.device) #self.scaling_multiplier[selected_pts_mask] ### TODO 找一个缩放方式
              
        self.densification_postfix(new_xyz, new_scaling_multiplier)

        self.n_gs = self.get_xyz.shape[0]
        
        
        # post process
        with torch.no_grad():
            self.reset_opacity()
            # self.reset_scales() #
            new_human_gs_out = self.forward()
        
        # reset scales
        interpolated_scales = attr['scales'][num_origin:]
        new_scales = new_human_gs_out['scales'][num_origin:].detach()
        scales_ratio = (interpolated_scales / new_scales.mean(-1)).detach()
        self.scaling_multiplier[num_origin:] *= scales_ratio.unsqueeze(1)
        self.scaling_multiplier = self.scaling_multiplier.detach()
        
        logger.info(f"Densification Ends")
        logger.info(f"Gaussian Number: {num_origin} -> {self.n_gs}")

    
    def prune_and_simplify(self, human_gs_out, opacity_threshold, scale_threshold, dead_grad=0.0005, large_scale=0.01, prune_max_n_gs_once=5000, min_n_gs=None, max_n_gs_once=None, exclude=None):
        
        logger.info(f"Pruning Begins")
        
        if self.n_gs <= min_n_gs:
            print('The total number of gaussians is near the minimum, skip pruning!')
            return
        
        opacity = human_gs_out['opacity'].clone()
        scales =  human_gs_out['scales'].clone()
        verts = human_gs_out['xyz_canon'].clone()
        edges = self._edges.clone()
        faces = self._faces.clone()

        
        # vertex blacklist: (transparent && small) || (dead && big)
        # verts_to_remove -> edges_unique

        # transparent small gaussians
        vert_mask = torch.where(opacity < opacity_threshold, True, False)
        vert_mask = torch.logical_and(vert_mask, scales[:, :1] < scale_threshold)
        print(f"### transparent small points, {vert_mask.sum()}")
        
        # dead large gaussians
        grads = self.xyz_gradient_accum / self.denom
        
        # only consider isotropic gaussians now
        dead_large_mask = scales[:, :1] > large_scale
        dead_large_mask = torch.logical_and(dead_large_mask, grads < dead_grad)
        print(f"### dead big points, {dead_large_mask.sum()}")
        
        vert_mask = torch.logical_or(vert_mask, dead_large_mask)
        
        # exclude non-decimated regions
        exclude = torch.tensor([6, 7]).long().to(self.device)
        if exclude is not None:
            non_decimate = torch.isin(self._vertex_label, exclude).unsqueeze(1)
            vert_mask = torch.logical_and(vert_mask, ~non_decimate)
            
            print(f'Give up the decimation in regions: {exclude}')
            print(f'Total Number: {non_decimate.sum()}')
        
        if vert_mask.sum() == 0:
            print('No selected points could be pruned')
            return
        
        if self.n_gs < min_n_gs:
            print('The total number of gaussians is below the minimum, skip pruning!')
            return 
        
        selected_vert_idx = torch.where(vert_mask)[0]
        face_mask = torch.isin(faces, selected_vert_idx).all(dim=1)
        # face_mask = torch.isin(faces, selected_vert).any(dim=1)

        selected_edges = faces[face_mask][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        edges = torch.sort(selected_edges, dim=1)[0]  # for undirected
        unique_edges, counts = torch.unique(edges, dim=0, return_counts=True)
        selected_edges = unique_edges[counts==2] # exclude boundaries
        
        # ## sort by opacity
        # edges_opacity = opacity.squeeze()[selected_edges]
        # sorted_opacity, indices = edges_opacity.sort(dim=1)
        # selected_edges = selected_edges.gather(1, indices)
        
        if selected_edges.shape[0] == 0:
            print('No selected edge could be collapse.')
            return
        elif selected_edges.shape[0] > 2 * prune_max_n_gs_once: # approximate value
            print('Too many edges selected, skip pruning this time.')
            return
        
        new_verts, new_faces, lbs_weight, prune_mask = collapse_edges(verts, self.lbs_weights, selected_edges, faces)
        
        ### TODO
        # subdivide again to fill the holes if the faces area is too large
        
        logger.info(f"Pruning Ends")
        logger.info(f"Gaussian Number: {self.n_gs} -> {new_verts.shape[0]}")   
        
        
        self.opacity_tmp = human_gs_out['opacity']
        self.scales_tmp = human_gs_out['scales_canon']
        self.rotmat_tmp = human_gs_out['rotmat_canon']
        self.prune_points(prune_mask)
        self.n_gs = self.get_xyz.shape[0]   
        torch.cuda.empty_cache()
        
        
        self.smpl_mesh = trimesh.Trimesh(vertices=new_verts.detach().cpu().numpy(), faces=new_faces.detach().cpu().numpy())
        unique_edges = self.smpl_mesh.edges_unique
        self._edges = torch.from_numpy(unique_edges.copy()).to(self.device).long()
        self._faces = new_faces
        self._mesh = Meshes(verts=[new_verts], faces=[new_faces])
        
        
        level_id = self._level_id[-1]
        self._level_id = self._level_id[:self.n_gs]
        self._vertex_label = self._vertex_label[~prune_mask]
        self.gs_level_mark[-1] = new_verts.shape[0]
        self.lbs_weights = lbs_weight
        
        # post process
        with torch.no_grad():
            self.reset_opacity()
            # skip scale, because of following densification
        
        
    def reset_opacity(self):

        tri_feats = self.triplane(self.get_xyz)

        curr_level = len(self.gs_level_mark) - 1
        opacity = torch.empty(0, 1, device=self.device)
        for i in range(curr_level):
            level_tri_feats = tri_feats[self.gs_level_mark[i]: self.gs_level_mark[i+1]]
            
            with torch.no_grad():
                self.appearance_dec_list[i].reset_opacity(level_tri_feats)
                opacity_min = torch.sigmoid(-self.appearance_dec_list[i].opacity_offset).min()
                appearance_out = self.appearance_dec_list[i](level_tri_feats)
            opacity = torch.vstack([opacity, appearance_out['opacity']])
            opacity_max = opacity.max()
        
        logger.info(f"Current opacity range: [{opacity_min:.6f}, {opacity_max:.6f}]")
        logger.info(f"Reset opacity...")
        logger.info(f"New opacity range: [0.5, {opacity_max:.6f}]")
         

    def reset_scales(self, large_scales=0.01, dead_grad=0.0005):
        tri_feats = self.triplane(self.get_xyz)

        curr_level = len(self.gs_level_mark) - 1
        scales = torch.empty(0, 3, device=self.device)
        for i in range(curr_level):
            level_tri_feats = tri_feats[self.gs_level_mark[i]: self.gs_level_mark[i+1]]
            
            with torch.no_grad():
                geometry_out = self.geometry_dec_list[i](level_tri_feats)
        
            scales = torch.vstack([scales, geometry_out['scales']])        
        
        grads = self.xyz_gradient_accum / self.denom
        dead_large_mask = scales[:, :1] > large_scales
        dead_large_mask = torch.logical_and(dead_large_mask, grads < dead_grad)
        mean_scales = scales[:, 0].mean()
        scales_ratio = (mean_scales / scales[:, 0]).detach()
        
        pts_above = scales[:, 0] > large_scales
        self.scaling_multiplier[pts_above] *= scales_ratio[pts_above].unsqueeze(1)
        self.scaling_multiplier = self.scaling_multiplier.detach()


        logger.info(f'Current scales range: [{scales.min():.6f}, {scales.max():.6f}]')
        logger.info(f"Reset scales...")
        logger.info(f'New scales range: [{scales.min():.6f}, {mean_scales:.6f}]')