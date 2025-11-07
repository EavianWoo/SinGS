import torch
import torch.nn as nn
import torch.nn.functional as F

from lpips import LPIPS

from .utils import l1_loss, ssim
from .loss_items import LaplacianSmoothing, L2Norm

from sings.rec.utils.image.sampler import PatchSampler
from sings.rec.utils.geometry.rotations import quaternion_to_matrix
from sings.rec.losses.loss_items import RegionLaplacianLoss_v2


class HumanLoss(nn.Module):
    def __init__(
        self,
        l_ssim_w=0.2,
        l_l1_w=0.8,
        l_lpips_w=0.0,
        num_patches=4,
        patch_size=32,
        use_patches=True,
        bg_color='white',
    ):
        super(HumanLoss, self).__init__()
        
        self.l_ssim_w = l_ssim_w
        self.l_l1_w = l_l1_w
        self.l_lpips_w = l_lpips_w
        self.use_patches = use_patches
        
        self.bg_color = bg_color
        self.lpips = LPIPS(net="vgg", pretrained=True).to('cuda')
    
        for param in self.lpips.parameters(): param.requires_grad=False
        
        if self.use_patches:
            self.patch_sampler = PatchSampler(num_patch=num_patches, patch_size=patch_size, ratio_mask=0.9, dilate=0)
        
    def forward(
        self, 
        data, 
        render_pkg,
        bg_color=None,
    ):
        loss_dict = {}
        extras_dict = {}
            
        gt_image = data['rgb']
        mask = data['mask'].unsqueeze(0)
        
        pred_img = render_pkg['render']
        
        gt_image = gt_image * mask + bg_color[:, None, None] * (1. - mask)
        extras_dict['gt_img'] = gt_image
        extras_dict['pred_img'] = pred_img

        
        if self.l_l1_w > 0.0:
            Ll1 = l1_loss(pred_img, gt_image, mask)
            
            loss_dict['l1'] = self.l_l1_w * Ll1

        if self.l_ssim_w > 0.0:
            loss_ssim = 1.0 - ssim(pred_img, gt_image)
            loss_ssim = loss_ssim * (mask.sum() / (pred_img.shape[-1] * pred_img.shape[-2]))
                
            loss_dict['ssim'] = self.l_ssim_w * loss_ssim
        
        if self.l_lpips_w > 0.0:
            if self.use_patches:
                bg_color_lpips = torch.rand_like(pred_img)
                image_bg = pred_img * mask + bg_color_lpips * (1. - mask)
                gt_image_bg = gt_image * mask + bg_color_lpips * (1. - mask)
                _, pred_patches, gt_patches = self.patch_sampler.sample(mask, image_bg, gt_image_bg)
 
                    
                loss_lpips = self.lpips(pred_patches.clip(max=1), gt_patches).mean()
                loss_dict['lpips_patch'] = self.l_lpips_w * loss_lpips
            else:
                bbox = data['bbox'].to(int)
                cropped_gt_image = gt_image[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                cropped_pred_img = pred_img[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                loss_lpips = self.lpips(cropped_pred_img.clip(max=1), cropped_gt_image).mean()
                loss_dict['lpips'] = self.l_lpips_w * loss_lpips
        
        loss = 0.0
        for k, v in loss_dict.items():
            loss += v
        
        return loss, loss_dict, extras_dict


class L2RegularizationLoss(nn.Module):
    def __init__(self,
        l2_norm_cfg=None,
        human_gs=None
    ):
        super().__init__()
        self._l_l2_norm = False

        
        if l2_norm_cfg is not None:
            self._l_l2_norm = True
            self.l2_norm = L2Norm(**l2_norm_cfg)
        
    def forward(self, human_gs_out, norm_list, nearest_edges=None):
        loss = 0.
        loss_dict = {}

        if self._l_l2_norm:
            norm_dict = {k: human_gs_out[k] for k in norm_list}
            loss_dict['l2'] = self.l2_norm(norm_dict)
            loss += loss_dict['l2']
        
        return loss, loss_dict


class LapRegularizationLoss(nn.Module):
    def __init__(self, type='standard', laplacian_cfg=None, human_gs=None):
        """pcd + knn"""
    
        """pcd mesh hybrid"""
        # self.pcd_laplacian = LaplacianSmoothing(K=16)
        return None

        if laplacian_cfg.position_strength > 0:
            self.mesh_laplacian_pos = RegionLaplacianLoss_v2(
                verts=human_gs.get_xyz, 
                edges=human_gs._edges,
                vertex_labels=human_gs._vertex_label,
                # faces=self.human_gs._faces,
                # laplacian_type='cotangent', 
                region_weights=laplacian_cfg.position_region_w
            )
        
        if laplacian_cfg.color_strength > 0:
            self.mesh_laplacian_color = RegionLaplacianLoss_v2(
                verts=human_gs.get_xyz,
                edges=human_gs._edges,
                vertex_labels=human_gs._vertex_label,
                region_weights=laplacian_cfg.position_region_w
            )
    
    def reset_laplacians(self, human_gs):
        
        self.mesh_laplacian_pos.reset_laplacians(human_gs)
        self.mesh_laplacian_color.reset_laplacians(human_gs)
        
    def forward(self, human_gs, alpha):
        
        pass
    