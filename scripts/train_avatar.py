import os
import sys
import time
import glob
import json
import argparse

from loguru import logger
from omegaconf import OmegaConf

sys.path.append('.')

from sings.rec.trainer.gs_trainer import SinGaussianTrainer
from sings.rec.defaults.config import cfg as default_cfg
from sings.rec.utils.general import safe_state


def get_logger(cfg):
    output_path = cfg.output_path
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    mode = 'eval' if cfg.eval else 'train'
    
    case_name = f"{cfg.dataset.name}"
    
    if cfg.dataset.batch is not None and cfg.dataset.batch != '':
        case_name = f"{cfg.dataset.batch}_{case_name}"
    if cfg.dataset.seq is not None and cfg.dataset.seq != '':
        case_name = f"{case_name}_{cfg.dataset.seq}"
    
    logdir = os.path.join(
        output_path, cfg.mode, 
        case_name,
        cfg.human.name, cfg.exp_name, 
        time_str,
    )

    cfg.logdir = logdir
    cfg.logdir_ckpt = os.path.join(logdir, 'ckpt')
    
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(cfg.logdir_ckpt, exist_ok=True)
    os.makedirs(os.path.join(logdir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(logdir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(logdir, 'anim'), exist_ok=True)
    os.makedirs(os.path.join(logdir, 'meshes'), exist_ok=True)
    
    logger.add(os.path.join(logdir, f'{mode}.log'), level='INFO')
    logger.info(f'Logging to {logdir}')
    logger.info(OmegaConf.to_yaml(cfg))
    
    with open(os.path.join(logdir, f'config_{mode}.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg)) 
    
    
def main(cfg):
    safe_state(seed=cfg.seed)
    
    get_logger(cfg)
    
    trainer = SinGaussianTrainer(cfg)
    
    if not cfg.eval:
        trainer.train()
        trainer.save_ckpt()
        trainer.save_splat()
    
    # run evaluation
    trainer.validate()
    
    mode = 'eval' if cfg.eval else 'train'
    with open(os.path.join(cfg.logdir, f'results_{mode}.json'), 'w') as f:
        json.dump(trainer.eval_metrics, f, indent=4)
    
    # run animation
    trainer.animate()
    trainer.render_canonical(pose_type='a_pose')
    trainer.render_canonical(pose_type='da_pose')


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg_file", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()
    
    cfg_file = OmegaConf.load(args.cfg_file)
    
    logger.info(f'Running experiment')
    
    default_cfg.cfg_file = args.cfg_file
    cfg = OmegaConf.merge(default_cfg, cfg_file, OmegaConf.from_cli(extras))
    main(cfg)
            