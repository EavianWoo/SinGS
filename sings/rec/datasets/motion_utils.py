# POSE_DIR
import numpy as np

import torch

from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle, quaternion_invert, quaternion_multiply
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


def manual_alignment(motion_type, motion_name=None):
    if motion_type == 'AMASS':
        manual_trans = np.array([0, 0, 10])
        manual_rot = np.array([90, 0, 0]) / 180 * np.pi
        manual_scale = 0.5
    
    elif motion_type == 'custom':
        manual_trans = np.array([0, 0, 0])
        manual_rot = np.array([-0.5, 0, 0]) / 180 * np.pi
        manual_scale = 1
    
    else:
        manual_trans = np.array([0, 0, 0])
        manual_rot = np.array([0, 0, 0]) / 180 * np.pi
        manual_scale = 0.5
    
    return manual_trans, manual_rot, manual_scale


def rebase_smpl(poses, transl, init_global_orient=None, init_transl=None):
    
    # reset global orient
    global_orient = poses[:, :3]
    
    mats = axis_angle_to_matrix(global_orient).float()  # (N, 3, 3)

    # compute inv rots[0]
    mat0_inv = torch.linalg.inv(mats[:1]).float()  # (3, 3)

    target_rotation = torch.tensor([torch.pi, 0, 0])
    mat_target = axis_angle_to_matrix(target_rotation)  # (4,)
    mats_new = (mat_target @ mat0_inv @ mats ).float() # (N, 4)
    
    # to aa
    rots_final = matrix_to_axis_angle(mats_new)  # (N, 3)
    
    transl = mat_target @ mat0_inv @ transl.view(-1, 3, 1).float()
    
    # reset translation
    transl = transl - transl[0, :]
    transl[:, -1] = transl[:, -1] + 20
    
    return poses, transl
