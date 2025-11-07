# AnimDataset: A dataset class for animation.
## 1 AMASS etc. Public Motion Dataset
## 2 Custom Motion Sequence Extracted from videos
### If you are interested in how to extract motions, please check the directory [playgroud/motions]

import os
import glob
from tqdm import tqdm

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

from loguru import logger

from sings.rec.defaults.constants import AMASS_SMPLH_TO_SMPL_JOINTS, ANIM_DIR
from sings.rec.utils.geometry import transformations
from sings.rec.utils.graphics import get_projection_matrix, get_projection_matrix_center, BasicPointCloud

from .Customdataset import CustomDataset
from .motion_utils import manual_alignment



class AnimDataset(Dataset):
    def __init__(
        self,
        motion_src='',
        motion_type='custom',
        motion_start=0,
        motion_end=200,
        motion_skip=4,
        render_size=(1024, 1024),
        image_zoom_ratio=1.,
        preload=True, 
        device='cuda',
    ):
        super().__init__()
        
        self.render_size = render_size
        self.image_zoom_ratio = image_zoom_ratio
        self.motion_name = os.path.basename(motion_src).split('.')[0]
        
        self.init_smpl(motion_src, motion_type, motion_start, motion_end, motion_skip)
        self.init_camera()
        
        seq = 'kunkun'
        manual_trans, manual_rot, manual_scale = manual_alignment(motion_type)
        manual_rotmat = transformations.euler_matrix(*manual_rot)[:3, :3]
        self.manual_rotmat = torch.from_numpy(manual_rotmat).float().unsqueeze(0)
        self.manual_trans = torch.from_numpy(manual_trans).float().unsqueeze(0)
        self.manual_scale = torch.tensor([manual_scale]).float().unsqueeze(0)
        
        
        self.num_frames = len(self.smpl_params['body_pose'])
        logger.info(f'Length of motion sequence: {self.num_frames}')

        self.cached_data = None
        self.device = device
        if preload:
            self.load_data_to_device(device)
            
    
    def __len__(self):
        return self.num_frames
    
    
    def init_camera(self, camera_path=None, fx=5000, fy=5000, znear=0.01, zfar=100.0):

        camera_extrinsic = torch.eye(4)

        world_view_transform = camera_extrinsic.to(torch.float32)
        c2w = torch.linalg.inv(camera_extrinsic).to(torch.float32)
        
        height = torch.tensor(self.render_size[0])
        wdith = torch.tensor(self.render_size[1])
        fovx = 2 * torch.arctan(wdith / (2 * fx))
        fovy = 2 * torch.arctan(height / (2 * fy))

        projection_matrix = get_projection_matrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        
        # if scale
        self.downscale = 1.0 / self.image_zoom_ratio
        if self.downscale > 1:
            self.render_size[0] = int(self.render_size[0] / self.downscale)
            self.render_size[1] = int(self.render_size[1] / self.downscale)
        
        self.datum = {}       
        self.datum.update({
            "fovx": fovx,
            "fovy": fovy,
            "image_height": self.render_size[0],
            "image_width": self.render_size[1],
            "world_view_transform": world_view_transform,
            "c2w": c2w,
            "full_proj_transform": full_proj_transform,
            "camera_center": camera_center,
        })
    
         
    def init_smpl(self, motion_src, motion_type, motion_start, motion_end, motion_skip, rebase_smpl=True): ### only need betas
        motions = np.load(motion_src)
        if motion_type == 'AMASS':
            
            poses = torch.from_numpy(motions['poses'][motion_start: motion_end: motion_skip, AMASS_SMPLH_TO_SMPL_JOINTS])
            transl = torch.from_numpy(motions['trans'][motion_start: motion_end: motion_skip])           
         
        elif motion_type == 'custom':
            
            poses = torch.from_numpy(motions['body_pose'][motion_start: motion_end: motion_skip])
            transl = torch.from_numpy(motions['transl'][motion_start: motion_end: motion_skip])

        
        if rebase_smpl:
            from .motion_utils import rebase_smpl
            poses, transl = rebase_smpl(poses, transl, ) #init_transl=, init_global_orient=)
        
        self.smpl_params = {
            'global_orient': poses[:, :3].float(),
            'body_pose': poses[:, 3:].float(),
            'transl': transl.view(-1, 3).float(), 
        }
    

    def load_data_to_device(self, device='cuda'):
        logger.info(f"Preloading {len(self)} frames to {device}...")
        smpl_params = {k: v.to(device) for k, v in self.smpl_params.items()}
        datum_base = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in self.datum.items()}

        self.cached_data = [
            {
                **datum_base,
                "global_orient": smpl_params["global_orient"][i],
                "body_pose": smpl_params["body_pose"][i],
                "transl": smpl_params["transl"][i],
                "smpl_scale": torch.ones(1, device=device),
                "manual_rotmat": self.manual_rotmat.to(device),
                "manual_scale": self.manual_scale.to(device),
                "manual_trans": self.manual_trans.to(device),
            }
            for i in range(self.num_frames)
        ]


    def __getitem__(self, idx):
        if self.cached_data is not None:
            return self.cached_data[idx]

    
    def get_chunk(self, start=None, end=None):
        """Return batched data."""
        if start is None:
            start = 0
        if end is None:
            end = self.num_frames

        smpl_params = {k: v.to(self.device) for k, v in self.smpl_params.items()}
        datum_base = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in self.datum.items()}
        
        ext_tfs = (self.manual_trans.expand(end - start, -1).to(self.device),
                   self.manual_rotmat.expand(end - start, -1, -1).to(self.device),
                   self.manual_scale.expand(end - start, -1).to(self.device))
                   

        sl = slice(start, end)
        out = {
            "global_orient": smpl_params["global_orient"][sl],
            "body_pose": smpl_params["body_pose"][sl],
            "transl": smpl_params["transl"][sl],
            "smpl_scale": torch.ones(end - start, 1, device=self.device),
            "ext_tfs": ext_tfs,
            **{k: v for k, v in datum_base.items() if isinstance(v, torch.Tensor)},
        }
        return out