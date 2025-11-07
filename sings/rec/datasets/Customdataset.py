import os
import glob
from tqdm import tqdm

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from loguru import logger

from sings.rec.utils.graphics import get_projection_matrix, get_projection_matrix_center
from sings.rec.defaults.constants import DATA_PATH



def load_smpl_param(path):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
    return {
        "betas": torch.from_numpy(smpl_params["betas"].astype(np.float32).reshape(1, -1)), # mean_beta
        "body_pose": torch.from_numpy(smpl_params["body_pose"].astype(np.float32)),
        "global_orient": torch.from_numpy(smpl_params["global_orient"].astype(np.float32)),
        "transl": torch.from_numpy(smpl_params["transl"].astype(np.float32)),
    }


def get_data_splits(img_list):
    data_length = len(img_list)
    # num_val = data_length // 5
    num_val = data_length // 10
    length = int(1 / (num_val) * data_length)
    offset = length // 2
    val_list = list(range(data_length))[offset::length]
    train_list = list(set(range(data_length)) - set(val_list))
    
    assert len(train_list) > 0 and len(val_list) > 0, f'Both lists for training and validation should not be empty!'

    return train_list, val_list


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self, batch, name, seq, split,
    ):
        
        data_path = os.path.join(DATA_PATH, batch) if batch is not None else DATA_PATH
        
        dataset_path = f"{data_path}/{name}/{seq}"
        
        self.prepare_everything(dataset_path, seq, split)
        
        
        if split == 'test': # test for ref image
            ### init test image
            ref_image_smpl_dir = os.path.join(dataset_path, "score_demo_image")
            self.init_ref_img(ref_image_smpl_dir)
        elif split == 'train+val':
            self.total_split = list(range(len(self.img_list)))
                    
        else:
            # Sometimes sam2 is not good at the early frames
            set_range = True
            start_idx = 2
            if set_range:
                self.img_list = self.img_list[start_idx:]
                self.msk_list = self.msk_list[start_idx:]
                self.smpl_params["body_pose"] = self.smpl_params["body_pose"][start_idx:]
                self.smpl_params["global_orient"] = self.smpl_params["global_orient"][start_idx:]
                self.smpl_params["transl"] = self.smpl_params["transl"][start_idx:]
            self.train_split, self.val_split = get_data_splits(self.img_list)
        
        self.split = split
        
        self.num_frames = len(self.smpl_params['body_pose'])

        self.cached_data = None
        if self.cached_data is None:
            self.load_data_to_cuda()

    def __len__(self):
        if self.split == "train":
            return len(self.train_split)
        elif self.split == "val":
            return len(self.val_split)
        elif self.split == "train+val":
            return len(self.total_split)
        # elif self.split == 'test':
        #     return 1
        # elif self.split == "anim":
        #     return self.num_frames
    
    def init_camera(self, camera_path, zfar=100.0, znear=0.01):
        camera = np.load(camera_path)
        K = camera["intrinsic"]
        camera_extrinsic = camera["extrinsic"]
        assert np.allclose(camera_extrinsic, np.eye(4))

        height = camera["height"]
        width = camera["width"]

        world_view_transform = torch.from_numpy(camera_extrinsic).to(torch.float32).T
        c2w = torch.from_numpy(np.linalg.inv(camera_extrinsic)).to(torch.float32)
             
        # Handle non-centered camera
        if abs(height // 2 - K[1, 2]) > 1.0 or abs(width // 2 - K[0, 2]) > 1.0:
            fov_left = np.arctan((K[0, 2]) / K[0, 0])
            fov_right = np.arctan((width - K[0, 2]) / K[0, 0])
            
            fov_top = np.arctan((K[1, 2]) / K[1, 1])
            fov_bottom = np.arctan((height - K[1, 2]) / K[1, 1])
            
            fovx = fov_left + fov_right
            fovy = fov_top + fov_bottom
            
            projection_matrix = get_projection_matrix_center(
                znear=znear, zfar=zfar, 
                fx=K[0, 0], fy=K[1, 1],
                cx=K[0, 2], cy=K[1, 2], 
                width=width, height=height).transpose(0,1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)         
            camera_center = world_view_transform.inverse()[3, :3]
            print('Using non-centered camera')

        else:
            fovx = 2 * np.arctan(width / (2 * K[0, 0]))
            fovy = 2 * np.arctan(height / (2 * K[1, 1]))

            projection_matrix = get_projection_matrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]
        
        # if scale
        self.downscale = 1.0 / self.image_zoom_ratio
        if self.downscale > 1:
            height = int(height / self.downscale)
            width = int(width / self.downscale)
            K[:2] /= self.downscale
        
        self.datum = {}       
        self.datum.update({
            "near": znear,
            "far": zfar,
            "fovx": fovx,
            "fovy": fovy,
            "image_height": height,
            "image_width": width,
            "world_view_transform": world_view_transform,
            "c2w": c2w,
            "full_proj_transform": full_proj_transform,
            "camera_center": camera_center,
            "cam_intrinsics": K,
        }
        )
               
    def init_smpl(self, smpl_path):
        smpl_params = load_smpl_param(smpl_path)

        smpl_params["body_pose"] = smpl_params["body_pose"]
        smpl_params["global_orient"] = smpl_params["global_orient"]
        smpl_params["transl"] = smpl_params["transl"]
        self.smpl_params = smpl_params
    
    def init_keypoints_2d(self, keypoints_path):
        self.keypoint_list = sorted(glob.glob(f"{keypoints_path}/*.json"))
        # self.keypoints_2d = np.load()
    
    def init_ref_img(self, ref_image_smpl_dir):
        ref_image_smpl_path = os.path.join(ref_image_smpl_dir, "input_pose.npz")
        ref_image_camera_path = os.path.join(ref_image_smpl_dir, "cameras.npz")

        self.init_camera(ref_image_camera_path)
        self.init_smpl(ref_image_smpl_path)
        pass
    
    def prepare_everything(
        self,
        data_root="./example/training_kits",
        video_name="SevenGod",
        split='train',
        image_zoom_ratio=1.0
    ):
        '''
        Preprocess data
        '''
        self.image_zoom_ratio = image_zoom_ratio

        root = os.path.join(data_root, video_name)

        self.img_list = sorted(glob.glob(f"{root}/images/*.png"))
        self.msk_list = sorted(glob.glob(f"{root}/masks/*.png"))
        
        assert len(self.img_list) == len(self.msk_list), \
            f"The length of image list and mask list should be the same. Got [{len(self.img_list)}] and [{len(self.msk_list)}]." 
        
        # use_parse = False
        # if os.path.exists(f"{root}/parsing"):
        #     self.parse_list = sorted(glob.glob(f"{root}/parsing/*.npy"))
        
        smpl_est_dir = os.path.join(root, "score_demo_video")
        smpl_path = os.path.join(smpl_est_dir, "poses_optimized.npz")
        if not os.path.exists(smpl_path):
            smpl_path = os.path.join(smpl_est_dir, "poses.npz")
            print(f"No optimized poses found, use original poses instead!")
            print(f"Load smpl parameters from {smpl_path}")
        
        camera_path = os.path.join(smpl_est_dir, "cameras.npz")
        
        self.init_camera(camera_path)
        self.init_smpl(smpl_path)
        
        if split == 'train+val':
            keypoints_path = os.path.join(root, 'keypoints_coco133', 'sapiens_0.6b')
            self.init_keypoints_2d(keypoints_path)


    def get_single_item(self, i):
        
        if self.split == "train":
            idx = self.train_split[i]
        elif self.split == "val":
            idx = self.val_split[i]
        elif self.split == 'train+val':
            idx = self.total_split[i]
        # elif self.split == "test": # test ref image
        #     idx = 0
        # elif self.split == "anim":
        #     idx = i
        
        datum = self.datum.copy()
        if self.split in ['train', 'val', 'train+val']:
            img = cv2.imread(self.img_list[idx])
            msk = cv2.imread(self.msk_list[idx], cv2.IMREAD_GRAYSCALE) / 255

            img = (img[..., :3][..., ::-1] / 255).astype(np.float32)
            img = img.transpose(2, 0, 1)
            msk = msk.astype(np.float32)
            
            datum.update({
                "rgb": torch.from_numpy(img).float(),
                "mask": torch.from_numpy(msk).float(),
            })
            
            # if hasattr(self, 'parse_list'):
            #     parse = np.load(self.parse_list[idx])
            #     selected_idx = [5, 14] # 3 - Hair, 
            #     # selected_parse = (parse == selected_idx[0]) | (parse == selected_idx[1]) ### to bool first, then * 1 to binary
            #     selected_parse = np.zeros_like(parse)
            #     for i in selected_idx:
            #         selected_parse = selected_parse | (parse == i)
            #     selected_parse = selected_parse * 1.0

            #     cv2.imwrite('./parse_valid/selected_parse.png',
            #                 selected_parse * 255)

            #     datum.update({
            #         "parse": torch.from_numpy(selected_parse).int(),
            #     })
        
        datum.update({
            "betas": self.smpl_params["betas"][0], ### temporal beta = mean_beta (10,)
            "global_orient": self.smpl_params["global_orient"][idx],
            "body_pose": self.smpl_params["body_pose"][idx],
            "transl": self.smpl_params["transl"][idx],
            "smpl_scale": torch.ones(1),
        })
        
        if  self.split == 'train+val':
            datum.update({
                'keypoint_path': self.keypoint_list[idx]
            })
        
        return datum

    def load_data_to_cuda(self):
        self.cached_data = []
        for i in tqdm(range(self.__len__())):
            datum = self.get_single_item(i)
            for k, v in datum.items():
                if isinstance(v, torch.Tensor):
                    datum[k] = v.to("cuda")
            self.cached_data.append(datum)

    def __getitem__(self, idx):
        if self.cached_data is None:
            return self.get_single_item(idx, is_src=True)
        else:
            return self.cached_data[idx]
