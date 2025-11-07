import os
import glob
import shutil
import itertools

import torch
import torchvision

import numpy as np
from PIL import Image
from tqdm import tqdm
from lpips import LPIPS
from loguru import logger
from omegaconf import OmegaConf

from pytorch3d.loss import mesh_edge_loss

from sings.rec.datasets.utils import (
    get_rotating_camera,
    get_smpl_canon_params,
    get_smpl_static_params, 
    get_static_camera
)
from sings.rec.losses.utils import ssim

from sings.rec.datasets.Customdataset import CustomDataset
from sings.rec.datasets.AnimDataset_opt import AnimDataset

from sings.rec.losses.loss import HumanLoss, L2RegularizationLoss, LapRegularizationLoss
from sings.rec.models.sings_hybrid import SinGS
# from sings.rec.models.sings_pose import SinGSPose

from sings.rec.renderer.gs_renderer_single import get_render_pkg
from sings.rec.utils.image.image import psnr, save_image
from sings.rec.utils.visualize.vis import save_ply, save_ellipsoid_meshes
from sings.rec.utils.general import RandomIndexIterator, save_images, create_video

# loss
from sings.rec.losses.loss_items import RegionLaplacianLoss_v2, GaussiansEdgeLoss


def get_train_dataset(cfg):
    logger.info(f'Loading {cfg.dataset.name} dataset {cfg.dataset.seq}-train')
    dataset = CustomDataset(cfg.dataset.batch, cfg.dataset.name, cfg.dataset.seq, 'train')
    
    return dataset


def get_val_dataset(cfg):
    logger.info(f'Loading {cfg.dataset.name} dataset {cfg.dataset.seq}-val')
    dataset = CustomDataset(cfg.dataset.batch, cfg.dataset.name, cfg.dataset.seq, 'val')
   
    return dataset


def get_ref_img_dataset(cfg):
    logger.info(f'Test reference image...')
    dataset = CustomDataset(cfg.dataset.batch, cfg.dataset.name, cfg.dataset.seq, 'test')
    return dataset


def get_anim_dataset(anim_cfg_path):
    anim_cfg = OmegaConf.load(anim_cfg_path)
    logger.info(f'Loading motion sequence from [{anim_cfg.motion_src}] for animation ...')
    dataset = AnimDataset(**anim_cfg)
        
    return dataset


class SinGaussianTrainer():
    def __init__(self, cfg, mode='train') -> None:
        self.cfg = cfg
        
        self.human_gs = None

        if cfg.bg_color == 'white':
            self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        elif cfg.bg_color == 'black':
            self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        else:
            raise ValueError(f"Unknown background color {cfg.bg_color}")
        
        # data
        if not cfg.eval:
            self.train_dataset = get_train_dataset(cfg)
        self.val_dataset = get_val_dataset(cfg)
        self.anim_dataset = get_anim_dataset(cfg.anim_cfg_path)
        
        # model
        if cfg.human.name == 'sings_hybrid':
            self.human_gs = SinGS(
                sh_degree=cfg.human.sh_degree, 
                isotropic=cfg.human.attribute_control.isotropic,
                fixed_opacity=cfg.human.attribute_control.fixed_opacity,
                init_opacity=cfg.human.attribute_control.init_opacity,
                init_scale_multiplier=cfg.human.attribute_control.init_scale_multiplier,
                thickness_factor=cfg.human.attribute_control.thickness_factor,
                refine_level=cfg.human.refine_level,
                
                body_template=cfg.human.body_template,
                n_subdivision=cfg.human.n_subdivision,  
                canonical_pose_type=cfg.human.canon_pose_type,
                kplanes_config=cfg.human.kplanes,
                n_features=cfg.human.feature_dim,
            )
            
        elif cfg.human.name == 'sings_vanilla':
            raise NotImplementedError
            if not cfg.eval:
                self.human_gs.initialize()
                self.human_gs = optimize_init(self.human_gs, num_steps=500)
                self.human_gs.reset_knns(K=16)
        
        if self.human_gs:
            logger.info(self.human_gs)
            if cfg.human.ckpt:
                self.human_gs.load_state_dict(torch.load(cfg.human.ckpt))
                logger.info(f'Loaded human model from {cfg.human.ckpt}')
            
            else:
                ckpt_files = sorted(glob.glob(f'{cfg.logdir_ckpt}/*human*.pth'))
                if len(ckpt_files) > 0:
                    ckpt = torch.load(ckpt_files[-1])
                    self.human_gs.load_state_dict(ckpt)
                    logger.info(f'Loaded human model from {ckpt_files[-1]}')
        
        if not cfg.eval:
            init_smpl_global_orient = torch.stack([x['global_orient'] for x in self.train_dataset.cached_data])
            init_smpl_body_pose = torch.stack([x['body_pose'] for x in self.train_dataset.cached_data])
            init_smpl_trans = torch.stack([x['transl'] for x in self.train_dataset.cached_data], dim=0)
            init_betas = torch.stack([x['betas'] for x in self.train_dataset.cached_data], dim=0)
            
            self.human_gs.create_global_orient(init_smpl_global_orient, cfg.human.optim_pose)
            self.human_gs.create_body_pose(init_smpl_body_pose, cfg.human.optim_pose)
            self.human_gs.create_transl(init_smpl_trans, cfg.human.optim_trans)
            self.human_gs.create_betas(init_betas[0], cfg.human.optim_betas)            

        else:
            init_betas = self.human_gs.betas
            
        self.human_gs.init_body_template(
            betas=self.human_gs.betas,
        )
            
        
        if cfg.eval:
            return 
        
        
        # optimizer
        self.human_gs.initialize()
        self.human_gs.init_attrs(init_steps=self.cfg.train.init_steps)
        self.human_gs.setup_optimizer(cfg=self.cfg.human.lr)
        
        
        # loss
        loss_cfg = cfg.human.loss
        self.loss_fn = HumanLoss(
            l_ssim_w=loss_cfg.ssim_w,
            l_l1_w=loss_cfg.l1_w,
            l_lpips_w=loss_cfg.lpips_w,
            num_patches=loss_cfg.num_patches,
            patch_size=loss_cfg.patch_size,
            use_patches=loss_cfg.use_patches,
            bg_color=self.bg_color,
        )
        
        self.l2_reg_loss_fn = L2RegularizationLoss(
            l2_norm_cfg=loss_cfg.l2_norm
        )
        
        # self.lap_reg_loss_fn = LapRegularizationLoss(
        #     laplacian_cfg=loss_cfg.laplacian
        # )
        
        if loss_cfg.laplacian.position_strength > 0.:
            self.region_mesh_lap_pos = RegionLaplacianLoss_v2(
                verts=self.human_gs.get_xyz, ## use original verts if standard laplacian
                edges=self.human_gs._edges,
                vertex_labels=self.human_gs._vertex_label,
                # faces=self.human_gs._faces,
                # laplacian_type='cotangent', 
                region_weights=loss_cfg.laplacian.position_regions_w
            )
        
        if loss_cfg.laplacian.color_strength > 0.:
            self.region_mesh_lap_color = RegionLaplacianLoss_v2(
                verts=self.human_gs.get_xyz, 
                edges=self.human_gs._edges,
                vertex_labels=self.human_gs._vertex_label,
                region_weights=loss_cfg.laplacian.color_regions_w
            )
        
        self.gaussian_connect_loss = GaussiansEdgeLoss()
        
        # metrics  
        self.eval_metrics = {}
        self.lpips = LPIPS(net="vgg", pretrained=True).to('cuda')
        

    def train(self):
        if self.human_gs:
            self.human_gs.train()

        pbar = tqdm(range(self.cfg.train.num_steps), desc="Training")
        
        rand_idx_iter = RandomIndexIterator(len(self.train_dataset))
        sgrad_means, sgrad_stds = [], []
        for t_iter in range(self.cfg.train.num_steps):            

            if hasattr(self.human_gs, 'update_learning_rate'):
                self.human_gs.update_learning_rate(t_iter)
        
            rnd_idx = next(rand_idx_iter)
            data = self.train_dataset[rnd_idx]
            
            human_gs_out = None
            
            if self.cfg.human.density_control.strategy == 'hybrid':
                clip_opacity_from = self.cfg.human.density_control.hybrid.prune_until_iter
            else:
                raise NotImplementedError
            
            
            opt_geo = False if t_iter < self.cfg.human.opt_geo_from and t_iter < self.cfg.human.opt_geo_until else True
            opt_app = False if t_iter < self.cfg.human.opt_app_from and t_iter < self.cfg.human.opt_app_until else True
            # clip_opacity = False if t_iter < clip_opacity_from else True # free opacity during pruning
            
            human_gs_out = self.human_gs.forward(
                smpl_scale=data['smpl_scale'], #[None],
                dataset_idx=rnd_idx,
                ext_tfs=None,
                opt_geo=opt_geo,
                opt_app=opt_app,
                clip_opacity=False,
            )
            
            bg_color = torch.rand(3, dtype=torch.float32, device="cuda")
            
            render_pkg = get_render_pkg(
                data=data, 
                human_gs_out=human_gs_out,
                bg_color=bg_color,
            )
            
            ### selected region
            # if 'selected_rendered_image' in render_pkg:
            #     selected_rendered_image = render_pkg['selected_rendered_image']
            #     selected_rendered_image = selected_rendered_image.permute(1, 2, 0).detach().cpu().numpy()


            impose_lap = True if t_iter > self.cfg.human.loss.laplacian.impose_from_iter else False
            loss_dict, reg_loss_dict, loss_extras = self._compute_losses(data, human_gs_out, render_pkg, bg_color, loss_cfg=self.cfg.human.loss, t_iter=t_iter, force_hand=True)

            
            if t_iter % 50 == 0:
                postfix_dict = {
                    "num_gs": f"{self.human_gs.n_gs/1000 :.1f}K",
                    # 'max_scale': human_gs_out['scales'][:, 0].max().data,
                }
                for k, v in loss_dict.items():
                    postfix_dict["l_"+k] = f"{v.item():.4f}"
                
                for k, v in reg_loss_dict.items():
                    postfix_dict["l_reg_"+k] = f"{v.item():.4f}"
                
                pbar.set_postfix(postfix_dict)
                pbar.update(50)
            
            # periodic check
            self._periodic_check(t_iter, loss_extras, human_gs_out)
             
            # density adjustment
            self._adjust_density(t_iter, render_pkg, human_gs_out)

        pbar.close()
        self._end_training()
    
    
    def _adjust_density(self, t_iter, render_pkg, human_gs_out):
        prune_flag = False
        if self.cfg.human.density_control.strategy == 'hybrid':
            self.density_cfg = self.cfg.human.density_control.hybrid
            
            # prune
            if t_iter >= self.density_cfg.prune_from_iter and \
            t_iter < self.density_cfg.prune_until_iter:
                
                rel_iter = t_iter - self.density_cfg.prune_from_iter
                if rel_iter % self.density_cfg.prune_interval == 0:
                    
                    prune_flag = True
                    render_pkg['human_viewspace_points'] = render_pkg['viewspace_points'][:human_gs_out['xyz'].shape[0]]
                    render_pkg['human_viewspace_points'].grad = render_pkg['viewspace_points'].grad[:human_gs_out['xyz'].shape[0]]
                    with torch.no_grad():
                        self.densify_or_prune_hybrid(
                            human_gs_out=human_gs_out,
                            visibility_filter=render_pkg['human_visibility_filter'],
                            radii=render_pkg['human_radii'],
                            viewspace_point_tensor=render_pkg['human_viewspace_points'],
                            mode='prune'
                        )
                                        
            # densify
            if t_iter >= self.density_cfg.densify_from_iter and \
            t_iter < self.density_cfg.densify_until_iter:
                
                rel_iter = t_iter - self.density_cfg.densify_from_iter - self.density_cfg.densify_interval
                if rel_iter % self.density_cfg.densify_interval == 0:
                    # recovery if just undergo prunning
                    if prune_flag:
                        print(f"will not densify right after prunning")
                        self.density_cfg.densify_interval += 1
                        prune_flag = False
                            
                    render_pkg['human_viewspace_points'] = render_pkg['viewspace_points'][:human_gs_out['xyz'].shape[0]]
                    render_pkg['human_viewspace_points'].grad = render_pkg['viewspace_points'].grad[:human_gs_out['xyz'].shape[0]]
                    with torch.no_grad():
                        self.densify_or_prune_hybrid(
                            human_gs_out=human_gs_out,
                            visibility_filter=render_pkg['human_visibility_filter'],
                            radii=render_pkg['human_radii'],
                            viewspace_point_tensor=render_pkg['human_viewspace_points'],
                            mode='densify'
                        )
        
        elif self.cfg.human.density_control.strategy == 'vanilla':
            self.density_cfg = self.cfg.human.density_control.vanilla
            if t_iter >= self.density_cfg.densify_from_iter and \
            t_iter < self.density_cfg.densify_until_iter:
                rel_iter = t_iter - self.density_cfg.densify_from_iter - self.density_cfg.densification_interval
                if rel_iter % self.density_cfg.densification_interval == 0:
                    render_pkg['human_viewspace_points'] = render_pkg['viewspace_points'][:human_gs_out['xyz'].shape[0]]
                    render_pkg['human_viewspace_points'].grad = render_pkg['viewspace_points'].grad[:human_gs_out['xyz'].shape[0]]
                    with torch.no_grad():
                        self.densifiy_and_prune_vanilla(
                            human_gs_out=human_gs_out,
                            visibility_filter=render_pkg['human_visibility_filter'],
                            radii=render_pkg['human_radii'],
                            viewspace_point_tensor=render_pkg['human_viewspace_points']
                        )
            
            self.reset_flag = False
    
    
    def _compute_losses(self, data, human_gs_out, render_pkg, bg_color, t_iter, loss_cfg, force_hand=True, impose_lap_from=1000):
        
        # Photometric
        loss, loss_dict, loss_extras = self.loss_fn(
            data,
            render_pkg,
            bg_color=bg_color,
        )
        
        # constrain attributes after density adjustment
        if t_iter >= self.cfg.human.density_control.hybrid.prune_until_iter and \
            t_iter >= self.cfg.human.density_control.hybrid.densify_until_iter:
            norm_list = ['xyz_offsets', 'scales', 'opacity']
        else:
            norm_list = ['xyz_offsets', 'scales']
        
        # regularization
        reg_loss, reg_loss_dict = self.l2_reg_loss_fn(human_gs_out, norm_list)
    
        # compactness
        smpl_mesh_edge_loss = loss_cfg.mesh_edge * mesh_edge_loss(self.human_gs._mesh)
        gaussian_connect_loss = loss_cfg.gaussian_connect * self.gaussian_connect_loss(human_gs_out)
        
        # smoothness
        num_level_0 = (self.human_gs._level_id == 0).sum()
        # region_mesh_lap_pt = self.region_mesh_lap(human_gs_out['xyz_canon'])
        region_mesh_lap_pos = self.region_mesh_lap_pos(human_gs_out['xyz_anchor_canon'][:num_level_0])
        region_mesh_lap_color = self.region_mesh_lap_color(human_gs_out['shs'][:num_level_0, 0])
        
        # linear increase
        lap_start = loss_cfg.laplacian.impose_from_iter
        alpha = 0.
        if t_iter > lap_start:
            alpha = loss_cfg.laplacian.position_strength * min(1, (t_iter - lap_start) / float(lap_start))
            
        beta = loss_cfg.laplacian.color_strength
        region_mesh_lap_pos_loss = alpha * region_mesh_lap_pos
        region_mesh_lap_color_loss = beta * region_mesh_lap_color
        
        
        if t_iter > 8000:
            alpha *= 2
        
        loss += smpl_mesh_edge_loss
        loss += gaussian_connect_loss
        loss += region_mesh_lap_pos_loss
        loss += region_mesh_lap_color_loss

        if force_hand:
            hand_lap = self.region_mesh_lap_pos.forward_hands(human_gs_out['xyz_canon'])
            loss += hand_lap * 1e-5
        
        
        loss += reg_loss   
        loss.backward()
        
        loss_dict['loss'] = loss
        loss_dict['mesh_edge_loss'] = smpl_mesh_edge_loss
        loss_dict['gaussian_connect_loss'] = gaussian_connect_loss
        
        reg_loss_dict['region_mesh_lap'] = region_mesh_lap_pos_loss
        reg_loss_dict['region_mesh_lap_color'] = region_mesh_lap_color_loss

        
        self.human_gs.optimizer.step()
        self.human_gs.optimizer.zero_grad(set_to_none=True)
        
        return loss_dict, reg_loss_dict, loss_extras
    
    
    def _update_progress(self, t_iter, freq=100):
        if t_iter % freq == 0:
            pass
    
    
    def _periodic_check(self, t_iter, loss_extras, human_gs_out):
        
        iter_s = 'final' if t_iter is None else f'{t_iter:06d}'
 
        # save checkpoint
        if (t_iter % self.cfg.train.save_ckpt_interval == 0) or \
            (t_iter == self.cfg.train.num_steps):
            self.save_ckpt(iter_s)

        # validation
        if t_iter % self.cfg.train.val_interval == 0 and t_iter > 0:
            self.validate(iter_s)
        
        # animation
        if self.anim_dataset is not None and t_iter % self.cfg.train.anim_interval == 0 and t_iter > 0:
            self.animate(iter_s)
        
        # visualize
        if t_iter % self.cfg.train.viz_interval == 0:
            self.visualize(iter_s, human_gs_out)
            self.render_canonical(iter_s, nframes=self.cfg.human.canon_nframes)
            
            with torch.no_grad():
                pred_img = loss_extras['pred_img']
                gt_img = loss_extras['gt_img']
                log_pred_img = (pred_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                log_gt_img = (gt_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                log_img = np.concatenate([log_gt_img, log_pred_img], axis=1)
                save_images(log_img, f'{self.cfg.logdir}/train/{iter_s}.png')
        
        
        # record training progress
        if self.cfg.train.save_progress_images and t_iter % self.cfg.train.progress_save_interval == 0:
            self.render_canonical(iter_s, nframes=2, is_train_progress=True)
        

        if t_iter % 1000 == 0 and t_iter > 0:
            if self.human_gs: self.human_gs.oneupSHdegree()
        
    
    def _end_training(self):
        # train progress images
        if self.cfg.train.save_progress_images:
            video_fname = f'{self.cfg.logdir}/train_{self.cfg.dataset.name}_{self.cfg.dataset.seq}.mp4'
            create_video(f'{self.cfg.logdir}/train_progress/', video_fname, fps=10)
            shutil.rmtree(f'{self.cfg.logdir}/train_progress/')
        
    
    def visualize(self, iter_s, human_gs_out, gs_pcd=True, gs_voxel=True):
        
        if gs_pcd == True:
            save_ply(human_gs_out, f'{self.cfg.logdir}/meshes/human_pcd_{iter_s}_splat.ply')
            
        if gs_voxel == True:
            save_ellipsoid_meshes(human_gs_out, f'{self.cfg.logdir}/meshes/human_voxel_{iter_s}')

         
    def save_ckpt(self, iter_s='final'):
            
        if self.human_gs:
            torch.save(self.human_gs.state_dict(), f'{self.cfg.logdir_ckpt}/human_{iter_s}.pth')
            
        logger.info(f'Saved checkpoint {iter_s}')


    def densify_or_prune_hybrid(self, human_gs_out, visibility_filter, radii, viewspace_point_tensor, mode='densify'):
        self.human_gs.max_radii2D[visibility_filter] = torch.max(
            self.human_gs.max_radii2D[visibility_filter], 
            radii[visibility_filter]
        )
        
        self.human_gs.add_densification_stats(viewspace_point_tensor, visibility_filter)


        if mode == 'prune':
            self.human_gs.prune_and_simplify(
                human_gs_out, 
                opacity_threshold=self.density_cfg.prune_opacity_threshold,
                scale_threshold=self.density_cfg.prune_scale_threshold,
                max_n_gs_once=self.density_cfg.prune_max_n_gs_once,
                min_n_gs=self.cfg.human.density_control.min_n_gaussians,
            )
        elif mode == 'densify':
            self.human_gs.densify_and_subdivide(
                human_gs_out,
                grad_threshold=self.density_cfg.densify_grad_threshold, 
                scale_threshold=self.density_cfg.densify_scale_threshold,
                max_screen_size=self.density_cfg.densify_render_size_threshold,
                max_n_gs=self.cfg.human.density_control.max_n_gaussians,
            )
        
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.region_mesh_lap_pos.reset_laplacians(verts=self.human_gs.get_xyz, 
                                              edges=self.human_gs._edges,
                                              vertex_labels=self.human_gs._vertex_label,
                                              faces=self.human_gs._faces)
        self.region_mesh_lap_color.reset_laplacians(verts=self.human_gs.get_xyz, 
                                                    edges=self.human_gs._edges,
                                                    vertex_labels=self.human_gs._vertex_label)
        

    def densifiy_and_prune_vanilla(self, human_gs_out, visibility_filter, radii, viewspace_point_tensor):
        self.human_gs.max_radii2D[visibility_filter] = torch.max(
            self.human_gs.max_radii2D[visibility_filter], 
            radii[visibility_filter]
        )
        
        self.human_gs.add_densification_stats(viewspace_point_tensor, visibility_filter)

        size_threshold = 20
        self.human_gs.densify_and_prune(
            human_gs_out,
            max_grad=self.density_cfg.densify_grad_threshold, 
            min_opacity=self.density_cfg.prune_min_opacity,
            densify_extent=self.density_cfg.densify_extent,
            percent_dense=self.density_cfg.percent_dense,
            max_screen_size=size_threshold,
            max_n_gs=self.cfg.human.density_control.max_n_gaussians,
        )
        
        self.human_gs.reset_knns(K=16) 
        
    
    @torch.no_grad()
    def validate(self, iter_s='final'):
                
        bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
        
        if self.human_gs:
            self.human_gs.eval()
                
        metrics = ['lpips', 'psnr', 'ssim']
        metrics = {k: [] for k in metrics}
        
        for idx, data in enumerate(tqdm(self.val_dataset, desc="Validation")):
            human_gs_out = None
            
            if self.human_gs:
                human_gs_out = self.human_gs.forward(
                    global_orient=data['global_orient'], 
                    body_pose=data['body_pose'], 
                    betas=data['betas'], 
                    transl=data['transl'], 
                    smpl_scale=data['smpl_scale'],
                    dataset_idx=-1,
                    ext_tfs=None,
                )
                    
            render_pkg = get_render_pkg(
                data=data, 
                human_gs_out=human_gs_out, 
                bg_color=bg_color,
            )
            
            gt_image = data['rgb']
            
            image = render_pkg["render"]
            if self.cfg.dataset.name == 'zju':
                image = image * data['mask']
                gt_image = gt_image * data['mask']
            
            metrics['psnr'].append(psnr(image, gt_image).mean().double())
            metrics['ssim'].append(ssim(image, gt_image).mean().double())
            metrics['lpips'].append(self.lpips(image.clip(max=1), gt_image).mean().double())
            
            log_img = torchvision.utils.make_grid([gt_image, image], nrow=2, pad_value=1)
            imf = f'{self.cfg.logdir}/val/full_{iter_s}_{idx:03d}.png'
            os.makedirs(os.path.dirname(imf), exist_ok=True)
            torchvision.utils.save_image(log_img, imf)
            
            log_img = []
            if len(log_img) > 0:
                log_img = torchvision.utils.make_grid(log_img, nrow=len(log_img), pad_value=1)
                torchvision.utils.save_image(log_img, f'{self.cfg.logdir}/val/human_{iter_s}_{idx:03d}.png')
        
        
        self.eval_metrics[iter_s] = {}
        
        for k, v in metrics.items():
            if v == []:
                continue
            
            logger.info(f"{iter_s} - {k.upper()}: {torch.stack(v).mean().item():.4f}")
            self.eval_metrics[iter_s][k] = torch.stack(v).mean().item()
        
        torch.save(metrics, f'{self.cfg.logdir}/val/eval_{iter_s}.pth')
    
    
    @torch.no_grad()
    def animate(self, iter_s='final', keep_images=True, save_splat=False):
        if self.anim_dataset is None:
            logger.info("No animation dataset found")
            return 0
        
        if self.human_gs:
            self.human_gs.eval()
        
        os.makedirs(f'{self.cfg.logdir}/anim/', exist_ok=True)
        
        human_gs_attrs = self.human_gs.get_gs_attrs()
        
        for idx, data in enumerate(tqdm(self.anim_dataset, desc="Animation")):
            human_gs_out = None
            
            if self.human_gs:
                
                ext_tfs = (data['manual_trans'], data['manual_rotmat'], data['manual_scale'])
                human_gs_out = self.human_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'] if 'betas' in data.keys() else self.human_gs.betas.detach(),
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'], #[None],
                    dataset_idx=-1,
                    ext_tfs=ext_tfs,
                    eval_mode=True,
                    gs_attrs=human_gs_attrs
                )
            
            human_gs_out['xyz'] = human_gs_out['xyz'].squeeze()
            render_pkg = get_render_pkg(
                data=data, 
                human_gs_out=human_gs_out, 
                bg_color=self.bg_color,
            )
            
            image = render_pkg["render"]
            
            torchvision.utils.save_image(image, f'{self.cfg.logdir}/anim/{idx:05d}.png')
            
            if save_splat:
                save_ply(human_gs_out, f'{self.cfg.logdir}/meshes/anim_{idx:05d}.ply', pose='deformed')
        
        video_fname = f'{self.cfg.logdir}/anim_{self.cfg.dataset.name}_{iter_s}.mp4'
        create_video(f'{self.cfg.logdir}/anim/', video_fname, fps=20)
        if not keep_images:
            shutil.rmtree(f'{self.cfg.logdir}/anim/')
            os.makedirs(f'{self.cfg.logdir}/anim/')
    
    
    @torch.no_grad()
    def animate_chunk(self, chunk_size=16, iter_s='final', save_video=True):
        if self.anim_dataset is None:
            logger.info("No animation dataset found")
            return 0
        num_frames = len(self.anim_dataset)
        
        if self.human_gs:
            self.human_gs.eval()
            
        os.makedirs(f'{self.cfg.logdir}/anim/', exist_ok=True)
        
        human_gs_attrs = self.human_gs.get_gs_attrs()
        
        import cv2
        import time
        start_time = time.time()
        
        for i in range(0, num_frames, chunk_size):
            batch_data = self.anim_dataset.get_chunk(i, i+chunk_size)
            
            human_gs_out = self.human_gs.forward_chunk(
                human_gs_attrs,
                global_orient=batch_data['global_orient'], 
                body_pose=batch_data['body_pose'], 
                betas=self.human_gs.betas.detach(), 
                transl=batch_data['transl'], 
                smpl_scale=batch_data['smpl_scale'],
                ext_tfs=batch_data['ext_tfs'],
                chunk_size=chunk_size,
            )
            
            for j in range(chunk_size):
                frame_idx = i + j
                
                if frame_idx >= num_frames:
                    break
                
                human_gs_out_frame = {}
                for k, v in human_gs_out.items():
                    if k == 'active_sh_degree':
                        human_gs_out_frame[k] = v
                    else:
                        human_gs_out_frame[k] = v[j]
                
                human_gs_out_frame['xyz'] = human_gs_out_frame['xyz'].squeeze()
                
                render_pkg = get_render_pkg(
                    data=self.anim_dataset[frame_idx], 
                    human_gs_out=human_gs_out_frame, 
                    bg_color=self.bg_color,
                )
                
                image = render_pkg["render"]
                img_np = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype('uint8')
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{self.cfg.logdir}/anim/{frame_idx:05d}.jpg', img_np)
                
        end_time = time.time()
        logger.info(f"Animation rendering time: {end_time - start_time:.2f}")
        logger.info(f"Total frames: {num_frames}")
        
        if save_video:
            video_fname = f'{self.cfg.logdir}/anim_{self.anim_dataset.motion_name}_{self.cfg.dataset.name}_{iter_s}.mp4'
            create_video(f'{self.cfg.logdir}/anim/', video_fname, fps=20, ext='jpg')
            print(f'Save video in [{video_fname}.mp4]')
    
    
    @torch.no_grad()
    def animate_more_without_render(self, iter_s='final', trans=None, rot=None, keep_images=False, anim_dataset=None, idx=0):
        if anim_dataset is None:
            logger.info("No animation dataset found")
            return 0
        
        if self.human_gs:
            self.human_gs.eval()
        
        data = anim_dataset[idx]
        ext_tfs = (data['manual_trans'], data['manual_rotmat'], data['manual_scale'])
        human_gs_out = self.human_gs.forward(
            global_orient=data['global_orient'],
            body_pose=data['body_pose'],
            betas=self.human_gs.betas.detach(),
            transl=data['transl'],
            smpl_scale=data['smpl_scale'],
            dataset_idx=-1,
            ext_tfs=ext_tfs,
            eval_mode=True,
        )
        
        human_gs_out['xyz'] = human_gs_out['xyz'].squeeze()
        return human_gs_out
        
    
    @torch.no_grad()
    def render_canonical(self, iter_s='final', nframes=100, is_train_progress=False, pose_type=None, keep_images=False):
        pose_type = pose_type if pose_type is not None else self.cfg.human.canon_pose_type
        
        if self.human_gs:
            self.human_gs.eval()
        
        os.makedirs(f'{self.cfg.logdir}/canon/', exist_ok=True)
        
        camera_params = get_rotating_camera(
            dist=5.0, img_size=256 if is_train_progress else 512, 
            nframes=nframes, device='cuda',
            angle_limit=torch.pi if is_train_progress else 2*torch.pi,
        )
        
        betas = self.train_dataset.cached_data[0]['betas']
        
        static_smpl_params = get_smpl_static_params(
            betas=betas,
            pose_type=pose_type,
        )
        
        if is_train_progress:
            progress_imgs = []
        
        pbar = range(nframes) if is_train_progress else tqdm(range(nframes), desc="Canonical:")
        
        human_gs_attrs = self.human_gs.get_gs_attrs()
        
        for idx in pbar:
            human_gs_out = None
            
            cam_p = camera_params[idx]
            data = dict(static_smpl_params, **cam_p)
            
            if self.human_gs:
                human_gs_out = self.human_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'],
                    dataset_idx=-1,
                    ext_tfs=None,
                    eval_mode=True,
                    gs_attrs=human_gs_attrs
                )
            
                
            if is_train_progress:
                scale_mod = 0.5
                render_pkg = get_render_pkg(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    bg_color=self.bg_color,
                    scaling_modifier=scale_mod,
                )
                
                image = render_pkg["render"]
                
                progress_imgs.append(image)
                
                render_pkg = get_render_pkg(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    bg_color=self.bg_color,
                )
                
                image = render_pkg["render"]
                
                progress_imgs.append(image)
                
            else:
                render_pkg = get_render_pkg(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    bg_color=self.bg_color,
                )
                
                image = render_pkg["render"]
                
                torchvision.utils.save_image(image, f'{self.cfg.logdir}/canon/{idx:05d}.png')
        
        if is_train_progress:
            os.makedirs(f'{self.cfg.logdir}/train_progress/', exist_ok=True)
            log_img = torchvision.utils.make_grid(progress_imgs, nrow=4, pad_value=0)
            save_image(log_img, f'{self.cfg.logdir}/train_progress/{iter:06d}.png', 
                       text_labels=f"{iter:06d}, n_gs={self.human_gs.n_gs}")
            return
        
        video_fname = f'{self.cfg.logdir}/canon_{pose_type}_{self.cfg.dataset.name}_{iter_s}.mp4'
        create_video(f'{self.cfg.logdir}/canon/', video_fname, fps=10)
        if not keep_images:
            shutil.rmtree(f'{self.cfg.logdir}/canon/')
            os.makedirs(f'{self.cfg.logdir}/canon/')
    
    
    def save_splat(self, predefined_pose='little_a_pose'):
        
        smpl_params = {}
        from sings.rec.datasets.utils import get_predefined_pose
        smpl_params['global_orient'] = torch.zeros(3, dtype=torch.float32, device="cuda")
        smpl_params['body_pose'] = get_predefined_pose(pose_type=predefined_pose)[0, :63]
        smpl_params['betas'] = self.human_gs.betas.detach()
        smpl_params['transl'] = torch.zeros(3, dtype=torch.float32, device="cuda")
        smpl_params['smpl_scale'] = torch.ones(1, dtype=torch.float32, device="cuda")

        human_gs_out = self.human_gs.forward(
            global_orient=smpl_params['global_orient'],
            body_pose=smpl_params['body_pose'],
            betas=smpl_params['betas'],
            transl=smpl_params['transl'],
            dataset_idx=-1,
            ext_tfs=None,
            eval_mode=True
        )
        
        save_ply(human_gs_out, f'{self.cfg.logdir}/showcase.ply', pose='deformed')
        
        logger.info('Save splat done.')
