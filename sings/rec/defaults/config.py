from omegaconf import OmegaConf

# general configuration
cfg = OmegaConf.create()
cfg.seed = 0
cfg.mode = 'human'
cfg.output_path = 'output'
cfg.cfg_file = ''
cfg.exp_name = 'test'
cfg.detect_anomaly = False
cfg.debug = False
cfg.wandb = False
cfg.logdir = ''
cfg.logdir_ckpt = ''
cfg.eval = False
cfg.bg_color = 'white'

# human dataset configuration
cfg.dataset = OmegaConf.create()
cfg.dataset.root_dir = ''
cfg.dataset.batch = ''
cfg.dataset.name = ''
cfg.dataset.seq = ''

# training configuration
cfg.train = OmegaConf.create()
cfg.train.batch_size = 1
cfg.train.num_workers = 0
cfg.train.num_steps = 30_000
cfg.train.save_ckpt_interval = 4000
cfg.train.val_interval = 2000
cfg.train.viz_interval = 2000
cfg.train.anim_interval = 2000
cfg.train.save_progress_images = False
cfg.train.progress_save_interval = 100

# human model configuration
cfg.human = OmegaConf.create()
cfg.human.name = 'sings_vanilla'
cfg.human.ckpt = None
cfg.human.sh_degree = 3
cfg.human.n_subdivision = 0
cfg.human.only_rgb = False
cfg.human.disable_posedirs = False

cfg.human.res_offset = False
cfg.human.rotate_sh = False

cfg.human.optim_pose = False
cfg.human.optim_betas = False
cfg.human.optim_trans = False
cfg.human.optim_eps_offsets = False
cfg.human.activation = 'relu'

cfg.human.canon_nframes = 60
cfg.human.canon_pose_type = 'da_pose'
cfg.human.body_template = 'smpl'

# human model learning rate configuration
cfg.human.lr = OmegaConf.create()
cfg.human.lr.position = 0.00016
cfg.human.lr.position_init = 0.00016
cfg.human.lr.position_final = 0.0000016
cfg.human.lr.position_delay_mult = 0.01
cfg.human.lr.position_max_steps = 30_000
cfg.human.lr.opacity = 0.05
cfg.human.lr.scaling = 0.005
cfg.human.lr.rotation = 0.001
cfg.human.lr.feature = 0.0025
cfg.human.lr.smpl_spatial = 2.0
cfg.human.lr.smpl_pose = 0.0001
cfg.human.lr.smpl_betas = 0.0001
cfg.human.lr.smpl_trans = 0.0001
cfg.human.lr.smpl_eps_offset = 0.0001

cfg.human.lr.appearance = 1e-3
cfg.human.lr.geometry = 1e-3
cfg.human.lr.vembed = 1e-3
cfg.human.lr.pose = 1e-3
cfg.human.lr.appearance_final = 1e-3
cfg.human.lr.geometry_final = 1e-3
cfg.human.lr.vembed_final = 1e-3
cfg.human.lr.pose_final = 1e-3
cfg.human.lr.mlp_max_steps = 16000


# human model loss coefficients
cfg.human.loss = OmegaConf.create()
cfg.human.loss.ssim_w = 0.2
cfg.human.loss.l1_w = 0.8
cfg.human.loss.lpips_w = 1.0
cfg.human.loss.num_patches = 4
cfg.human.loss.patch_size = 128
cfg.human.loss.use_patches = 1


# regularization
cfg.human.loss.laplacian = OmegaConf.create()
cfg.human.loss.laplacian.type = 'standard'
cfg.human.loss.laplacian.impose_on = ['postition', 'color']
cfg.human.loss.laplacian.strength = 500
cfg.human.loss.laplacian.regional = True
cfg.human.loss.l2_norm = 0.005


# density control
cfg.human.density_control = OmegaConf.create()
cfg.human.density_control.strategy = 'hybrid'

cfg.human.density_control.vanilla = OmegaConf.create()
cfg.human.density_control.vanilla.densification_interval = 1000
cfg.human.density_control.vanilla.densify_from_iter = 999
cfg.human.density_control.vanilla.densify_until_iter = 15_000
cfg.human.density_control.vanilla.prune_min_opacity = 0.005
cfg.human.density_control.vanilla.densify_extent = 1.0
cfg.human.density_control.vanilla.percent_dense = 0.01

cfg.human.density_control.hybrid = OmegaConf.create()
cfg.human.density_control.hybrid.densify_interval = 2000
cfg.human.density_control.hybrid.densify_from_iter = 1999
cfg.human.density_control.hybrid.densify_until_iter = 12000
cfg.human.density_control.hybrid.densify_grad_threshold = 0.001
cfg.human.density_control.hybrid.densify_scale_threshold = 0.01
cfg.human.density_control.hybrid.densify_render_size_threshold = 20

cfg.human.density_control.hybrid.prune_interval = 2000
cfg.human.density_control.hybrid.prune_from_iter = 1999
cfg.human.density_control.hybrid.prune_until_iter = 12000
cfg.human.density_control.hybrid.prune_opacity_threshold = 0.005
cfg.human.density_control.hybrid.prune_collapse_rate = 0.5

cfg.human.density_control.max_n_gaussians = 2e5
cfg.human.density_control.min_n_gaussians = 1e5


# gaussian attributes control
cfg.human.attribute_control = OmegaConf.create()
cfg.human.attribute_control.isotropic = True
cfg.human.attribute_control.thickness_factor = 1.0

cfg.human.attribute_control.fixed_opacity = False
cfg.human.attribute_control.init_opacity = 0.8
cfg.human.attribute_control.init_scale_multiplier = 0.8
cfg.human.attribute_control.clip_opacity_from = 12000
cfg.human.attribute_control.os_reset_interval = 2000

# decoder optimization control
cfg.human.opt_geo_from = 1000
cfg.human.opt_geo_until = 14000
cfg.human.opt_app_from = 1000
cfg.human.opt_app_until = 15000


