import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))


SMPL_PATH = os.path.join(current_dir, '../../..', 'data/human_models/smpl')
SMPLH_PATH = os.path.join(current_dir, '../../..', 'data/human_models/smplh')


DATA_PATH = os.path.join(current_dir, '../../..', 'examples/training_kits')
ANIM_DIR = os.path.join(current_dir, '../../..', 'data/animation')

# map amass to smpl
AMASS_SMPLH_TO_SMPL_JOINTS = np.arange(0,156).reshape((-1,3))[[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 
    19, 20, 21, 22, 37
]].reshape(-1)