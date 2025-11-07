'''
Optimize SMPL+H parameters with keypoints from Sapiens.
Optimization objective: mse.
'''

import os
import sys
sys.path.append('..')
sys.path.append('../../')

import json
import joblib
import argparse
from tqdm import tqdm

import numpy as np
import torch

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, PerspectiveCameras
)

from smplx import SMPL as smpl

from sings.rec.utils.body_model.smpl import SMPL
from sings.rec.utils.body_model.smplh import SMPLH
from sings.rec.utils.geometry import pcd_projector

from sings.rec.datasets.Customdataset import CustomDataset


def coco17_to_smpl(coco2d):
    '''
    input 2d joints in coco dataset format,
    and out 2d joints in SMPL format.
    Non-overlapping joints are set to 0s. 
    '''
    assert coco2d.shape == (17, 2)
    smpl2d = np.zeros((24, 2))
    smpl2d[1]  = coco2d[11] # leftUpLeg
    smpl2d[2]  = coco2d[12] # rightUpLeg
    smpl2d[4]  = coco2d[13] # leftLeg
    smpl2d[5]  = coco2d[14] # rightLeg
    smpl2d[7]  = coco2d[15] # leftFoot
    smpl2d[8]  = coco2d[16] # rightFoot
    smpl2d[16] = coco2d[5]  # leftArm
    smpl2d[17] = coco2d[6]  # rightArm
    smpl2d[18] = coco2d[7]  # leftForeArm
    smpl2d[19] = coco2d[8]  # rightForeArm
    smpl2d[20] = coco2d[9]  # leftHand
    smpl2d[21] = coco2d[10] # rightHand
    return smpl2d

# def select_joints():
#     ankle = [15, 16]
#     left_feet_index = [17, 18, 19]
#     right_feet_index = [20, 21, 22]


def coco133_to_smplh(coco2d, selected_keypoints=None):
    '''
    For hands and feet refinement.
    '''
    assert coco2d.shape == (133, 2)
    
    # enable extra joints of smpl
    smplh2d = torch.zeros((73, 2))
    
    # Only optimize arm, leg [elbow, wrist, ankle and feet]
    
    # elbow
    smplh2d[18] = coco2d[7]
    smplh2d[19] = coco2d[8]
    
    # wrist
    smplh2d[20] = coco2d[9]
    smplh2d[21] = coco2d[10]
    
    # knee
    smplh2d[4] = coco2d[13]
    smplh2d[5] = coco2d[14]
    
    # ankle
    smplh2d[7] = coco2d[15]
    smplh2d[8] = coco2d[16]
    
    # left foot
    smplh2d[57] = coco2d[17] # 18 19 20 -> leftFeet
    smplh2d[58] = coco2d[18]
    smplh2d[59] = coco2d[19]
    
    # right foot
    smplh2d[60] = coco2d[20] # 21 22 23 -> rightFeet
    smplh2d[61] = coco2d[21]
    smplh2d[62] = coco2d[22]
    
    # left hand root
    smplh2d[20] = coco2d[91]
    smplh2d[21] = coco2d[112]
    
    # right hand
    smplh2d[51] = coco2d[115] # right thumb 3
    smplh2d[39] = coco2d[119] # right fore finger 3
    smplh2d[42] = coco2d[123] # right middle finger 3
    smplh2d[45] = coco2d[127] # right ring finger 3
    smplh2d[48] = coco2d[131] # right pinky finger 3
    
    # left hand
    smplh2d[36] = coco2d[94] # thumb 3
    smplh2d[24] = coco2d[98] # fore finger 3
    smplh2d[27] = coco2d[102] # middle finger 3
    smplh2d[30] = coco2d[106] # ring finger 3
    smplh2d[33] = coco2d[110] # pinky finger 3
    
    return smplh2d


# def coco133hand_to_smplh(coco2d):
    
#     # left hand
#     smplh2d[26] = coco2d[94] # thumb 3
#     smplh2d[29] = coco2d[98] # fore finger 3
#     smplh2d[32] = coco2d[102] # middle finger 3
#     smplh2d[35] = coco2d[106] # ring finger 3
#     smplh2d[38] = coco2d[110] # pinky finger 3
    
#     # right hand


def silhouette_renderer_from_pinhole_cam(cam, device='cpu'):
    
    focal_length = torch.tensor([[cam['cam_intrinsics'][0, 0], cam['cam_intrinsics'][1, 1]]]).float()
    principal_point = torch.tensor([[cam['image_width'] - cam['cam_intrinsics'][0, 2], 
                                     cam['image_height'] - cam['cam_intrinsics'][1, 2]]]).int()  # In PyTorch3D, we assume that +X points left, and +Y points up and +Z points out from the image plane.
    # image_size = torch.tensor([[cam['image_height'], cam['image_width']]])
    image_size = torch.from_numpy(np.array([[cam['image_height'], cam['image_width']]])).int()
    cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point, in_ndc=False, image_size=image_size, device=device)
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(
        image_size=(cam['image_height'].item(), cam['image_width'].item()),
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=150, ### RuntimeError: Must have num_closest <= 150
    )
    
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    return silhouette_renderer


def to_homogeneous(pts):
    if isinstance(pts, torch.Tensor):
        return torch.cat([pts, torch.ones_like(pts[..., 0:1])], axis=-1)
    elif isinstance(pts, np.ndarray):
        return np.concatenate([pts, np.ones_like(pts[..., 0:1])], axis=-1)


def to_tensor(array, dtype=torch.float32, device=torch.device('cpu')):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype).to(device)
    else:
        return array.to(device)


def vertext_forward(pose, betas, body_model, scale):
    device = pose.device

    T_pose = torch.zeros_like(pose)
    T_pose = T_pose.reshape(-1, 3)
    
    _, mesh_transf = body_model.verts_transformations(
        return_tensor=True,
        poses=pose[None],
        betas=betas[None],
        transl=torch.zeros([1, 3]).float().to(device),
        concat_joints=True
    )
    
    s = torch.eye(4).to(mesh_transf.device)
    s[:3, :3] *= scale
    mesh_transf = s @ mesh_transf
    
    posed_verts, posed_joints = body_model(
        return_tensor=True,
        return_joints=True,
        poses=pose[None],
        betas=betas[None],
        transl=torch.zeros([1, 3]).float().to(device),
    )
    
    return posed_verts, posed_joints


def turn_smpl_gradient_on(select=['up', 'down']):
    '''
    only apply gradients on assigned joints.
    '''
    grad_mask = np.zeros([73, 3])   ### CHECK
    
    visible = []
    if 'up' in select:
        visible += ['Elbow', 'wrist', 'hand']
    if 'down' in select:
        visible += ['Knee', 'Ankle', 'Big Toe', 'Little Toe', 'Heel']
    # if 'Upper Leg Left' in visible:
    #     grad_mask[1, 0] = 1
    #     grad_mask[1, 2] = 1
    # if 'Upper Leg Right' in visible:
    #     grad_mask[2, 0] = 1
    #     grad_mask[2, 2] = 1
    # if 'Lower Leg Left' in visible:
    #     grad_mask[4, 0] = 1
    # if 'Lower Leg Right' in visible:
    #     grad_mask[5, 0] = 1
    # if 'Left Foot' in visible:
    #     grad_mask[7] = 1
    # if 'Right Foot' in visible:
    #     grad_mask[8] = 1
    # if 'Upper Arm Left' in visible:
    #     grad_mask[16, 1] = 1
    #     grad_mask[16, 2] = 1
    # if 'Upper Arm Right' in visible:
    #     grad_mask[17, 1] = 1
    #     grad_mask[17, 2] = 1
    # if 'Lower Arm Left' in visible:
    #     grad_mask[18, 1] = 1
    # if 'Lower Arm Right' in visible:
    #     grad_mask[19, 1] = 1
    
    if 'Elbow' in visible:
        grad_mask[18] = 0.5
        grad_mask[19] = 0.5
    if 'Wrist' in visible:
        grad_mask[20] = 0.5
        grad_mask[21] = 0.5 
    # if 'hand' in visible:
        
    # for feet
    if 'Knee' in visible:
        grad_mask[4] = 1
        grad_mask[5] = 1
    if 'Ankle' in visible:
        grad_mask[7] = 1
        grad_mask[8] = 1
    if 'Big Toe' in visible: 
        grad_mask[29] = 1
        grad_mask[32] = 1
    if 'Little Toe' in visible: 
        grad_mask[30] = 1
        grad_mask[33] = 1
    if 'Heel' in visible:
        grad_mask[31] = 1
        grad_mask[34] = 1
    return grad_mask.reshape(-1)


def clip_smpl_vals():
    '''
    limit the pose(joint angle) changes for certain joints.
    for example, knee only has 1 degree of freedom.(in skeleton model)
    '''
    limits = np.ones([24, 3, 2])
    limits[..., 0] *= -360
    limits[..., 1] *= 360
    # knees
    limits[4, 0] = [0, 160]
    limits[4, 1] = [0, 0]
    limits[4, 2] = [0, 0]
    limits[5, 0] = [0, 160]
    limits[5, 1] = [0, 0]
    limits[5, 2] = [0, 0]
    # feet
    limits[7, 0] = [-45, 90]
    limits[7, 1] = [-60, 60]
    limits[7, 2] = [-10, 10]
    limits[8, 0] = [-45, 90]
    limits[8, 1] = [-60, 60]
    limits[8, 2] = [-10, 10]
    # elbow
    limits[18, 1] = [-160, 0]
    limits[19, 2] = [0, 160]
    
    
    return limits.reshape(-1, 2) / 180 * np.pi


def optimize_smpl(camera, data_item, body_model, scale=1.0, num_iters=50):
    device = body_model.device
    
    torch.set_default_dtype(torch.float32)
    
    # create renderer
    renderer = silhouette_renderer_from_pinhole_cam(camera, device=device)
    
    R = camera["world_view_transform"][:3, :3][None].float().to(device)
    T = data_item['transl'][None].float().to(device)

    # create tensors
    smpl_betas = data_item['betas']
    hand_betas = torch.zeros(6)
    smpl_betas = torch.cat([smpl_betas, hand_betas])
    
    smpl_poses = torch.concat([data_item['global_orient'], data_item['body_pose']]) # 1 + 23
    
    pose = torch.nn.Parameter(smpl_poses.float().to('cuda'), requires_grad=True)
    betas = torch.nn.Parameter(smpl_betas.float().to('cuda'), requires_grad=True)
    
    mask_target = data_item['mask'].to(device)
    
    with open(data_item['keypoint_path'], 'r') as f:
        keypoints = json.load(f)
    
    keypoints_2d = torch.tensor(keypoints['instance_info'][0]['keypoints']).to('cuda')
    keypoints_score = torch.tensor(keypoints['instance_info'][0]['keypoint_scores']).to('cuda')
    
    
    joints_target = keypoints_2d
    joints_target[keypoints_score < 0.8] = 0 # discard low confidence detections
    
    joints_target = coco133_to_smplh(joints_target[:, :2]).float().to(device)
    joints_mask = (joints_target.sum(dim=1) != 0)

    # create optimizer
    optim_list = [{"params": pose, "lr": 2e-3}, # lr 5e-3
                  {"params": betas, "lr": 2e-3}]
    optim = torch.optim.Adam(optim_list)

    # only allow gradient w.r.t certain joints
    grad_mask = turn_smpl_gradient_on()
    grad_mask = torch.from_numpy(grad_mask).float().to(device)

    limits = clip_smpl_vals()
    limits = torch.from_numpy(limits).float().to(device)

    # for i in tqdm(range(num_iters), total=num_iters):
    for i in range(num_iters):
        optim.zero_grad()
        
        world_verts, world_joints = vertext_forward(pose, betas, body_model, scale)
        
        smpl_faces = body_model.faces_tensor.long()
        
        mesh = Meshes(
            verts=[world_verts],
            faces=[smpl_faces]
        )
        
        # ### check obj
        # from pytorch3d.io import save_obj
        # save_obj('./viz_debug/test_smpl.pbj',
        #          verts=mesh.verts_packed(), faces=mesh.faces_packed())
        
        camera_extrinsic = torch.eye(4)
        camera_extrinsic[:3, :3] = R
        camera_extrinsic[:3, 3] = T
        
        proj_joints = pcd_projector.pcd_3d_to_pcd_2d_torch(
            (world_joints.T)[None],
            torch.from_numpy(camera['cam_intrinsics']).float().to(device)[None],
            camera_extrinsic.float().to(device)[None],
            torch.from_numpy(np.array([camera['image_height'], camera['image_width']])).float().to(device)[None], #
            keep_z=False,
            norm_coord=False,
        )[0].T

        loss = torch.nn.functional.mse_loss(
            proj_joints[joints_mask], joints_target[joints_mask]
        ) / joints_mask.sum()
       
        silhouette = renderer(meshes_world=mesh, R=R, T=T)
        silhouette = torch.rot90(silhouette[0, ..., 3], k=2)
        
        # ### check silhouette
        # import cv2
        # silhouette_np = (silhouette.detach().cpu().numpy() * 255).astype(np.uint8)
        # cv2.imwrite('./viz_debug/test_silhouette_smlph.png', silhouette_np)
        
        # silhouette_np = np.repeat(silhouette_np[..., None], 3, axis=-1)
        # for joint, proj_joint in zip(joints_target, proj_joints):
        #     if joint[0] > 0 and joint[1] > 0:
        #         cv2.circle(silhouette_np, tuple(joint.int().cpu().numpy()), radius=5, color=(0, 255, 0), thickness=-1)
        #         cv2.circle(silhouette_np, tuple(proj_joint.int().cpu().numpy()), radius=5, color=(0, 0, 255), thickness=-1)
        # cv2.imwrite('./viz_debug/test_silhouette_smplh_annot.png', silhouette_np)
        
        loss += torch.nn.functional.mse_loss(silhouette, mask_target)
        
        loss.backward()
        valid_mask = ((pose < limits[..., 1]) * (pose > limits[..., 0])).float()
        
        pose.grad = pose.grad * grad_mask[:72] * valid_mask
        betas.grad[:10] = 0 # clear body shape backward
        
        if torch.isnan(betas.grad).any():
            print('Warning: grad anomaly occurs.')
            continue
        
        optim.step()
    return pose.detach(), betas.detach()


def main(kit_dir, num_iters = 50):
    device = torch.device('cuda')
    
    body_model = SMPLH(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/human_models/smplh'),
        gender='neutral',
        device=device,
    )
    
    smpl_betas = []
    smpl_poses = []
    # smpl_trans = []
    
    KIT_DIR = kit_dir
    batch_dir, name = os.path.split(kit_dir)
    batch = os.path.basename(batch_dir.strip('/'))
    
    dataset = CustomDataset(batch, name, "", "train+val")
    data_length = len(dataset.img_list)
    
    print(f'Set Optimization Iteration = [{num_iters}] for each frame.')
    for i in tqdm(range(data_length), total=data_length, desc='SMPLH Optimization'):
        data_item = dataset.get_single_item(i)
        pose_item, betas_item = optimize_smpl(
            dataset.datum,
            data_item,
            body_model,
            num_iters=num_iters
        )
        smpl_betas.append(betas_item.unsqueeze(0))
        smpl_poses.append(pose_item.unsqueeze(0))

    smpl_betas = torch.cat(smpl_betas)
    smpl_poses = torch.cat(smpl_poses)
    # smpl_trans = torch.cat()
    
    pose_dict = {
        'betas': smpl_betas.mean(0).cpu().numpy(),
        'global_orient': smpl_poses[:, :3].reshape(-1, 3).cpu().numpy(),
        'body_pose': smpl_poses[:, 3:].reshape(-1, 69).cpu().numpy(),
        'transl':  dataset.smpl_params['transl']
    }
    

    save_path = os.path.join(KIT_DIR, 'score_demo_video')
    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(os.path.join(save_path, "poses_optimized.npz"), **pose_dict)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kit_dir', type=str, default=None, required=True)
    parser.add_argument('--num_iters_per_frame', type=int, default=50, required=False)
    args = parser.parse_args()
    main(args.kit_dir, args.num_iters_per_frame)