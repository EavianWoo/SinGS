# In case of issues with the PHALP's downloading of SMPL model files,
# this patch fixes it by copying existing files.

import os
import shutil

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path,  "../submodules/ScoreHMR"))
from phalp.configs.base import CACHE_DIR


src_smpl_path = os.path.join(os.path.join(dir_path, "../../data/human_models/smpl/SMPL_NEUTRAL.pkl"))
tgt_smpl_path = os.path.join(CACHE_DIR, "phalp/3D/models/smpl/SMPL_NEUTRAL.pkl")
shutil.copy(src_smpl_path, tgt_smpl_path)
