#!/bin/bash
# This script downloads for the preprocessing.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ENV_NAME="sings"

echo -e "\033[31mPreparing necessary data for preprocessing tools...\033[0m"

# alphapose
## files to download:
## 1) halpe26_fast_res50_256x192.pth
##    https://drive.google.com/uc?id=1S-ROA28de-1zvLv-hVfPFJ5tFBYOSITb
##    -> [$ROOT_DIR/preprocess/submodules/AlphaPose/pretrained_models/]
## 2) yolov3-spp.weights
##    https://drive.google.com/uc?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC
##    -> [$ROOT_DIR/preprocess/submodules/AlphaPose/detector/yolo/data/]
prepare_alphapose() {
    CURRENT_STAGE="AlphaPose"
    echo -e "\033[34m[${CURRENT_STAGE}] Downloading [halpe26_fast_res50_256x192.pth]...\033[0m"
    gdown 1S-ROA28de-1zvLv-hVfPFJ5tFBYOSITb -O $ROOT_DIR/preprocess/submodules/AlphaPose/pretrained_models/halpe26_fast_res50_256x192.pth
    echo -e "\n"

    echo -e "\033[34m[${CURRENT_STAGE}] Downloading [yolov3-spp.weights]...\033[0m"
    mkdir $ROOT_DIR/preprocess/submodules/AlphaPose/detector/yolo/data
    gdown 1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC -O $ROOT_DIR/preprocess/submodules/AlphaPose/detector/yolo/data/yolov3-spp.weights

    echo -e "\033[32m[${CURRENT_STAGE}] Done.\033[0m"
}

# sam2
## files to download:
## - sam2.1_hiera_large.pt
##   https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
##   -> [$ROOT_DIR/preprocess/submodules/sam2/checkpoints/]
prepare_sam2() {
    CURRENT_STAGE="SAM2"
    echo -e "\033[34m[${CURRENT_STAGE}] Downloading [sam2.1_hiera_large.pt]...\033[0m"
    wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
        -O preprocess/submodules/sam2/checkpoints/sam2.1_hiera_large.pt
    echo -e "\n"

    # mv sam2 scripts
    echo -e "\033[34m[${CURRENT_STAGE}] Moving [$ROOT_DIR/preprocess/utils/get_masks_for_sings.py] to [$ROOT_DIR/preprocess/submodules/sam2/]...\033[0m"
    cp $ROOT_DIR/preprocess/utils/get_masks_for_sings.py $ROOT_DIR/preprocess/submodules/sam2/

    echo -e "\033[32m[${CURRENT_STAGE}] Done.\033[0m"
    echo -e "\n"
}

# scorehmr
## files to download:
## - ScoreHMR official data
##   https://drive.google.com/uc?id=1W53UMg8kee3HGRTNd2aNhMUew_kj36OH
##   -> [$ROOT_DIR/preprocess/submodules/ScoreHMR/] -> unzip
prepare_scorehmr() {
    CURRENT_STAGE="ScoreHMR"
    echo -e "\033[34m[${CURRENT_STAGE}] Downloading [ScoreHMR official data]...\033[0m"
    cd $ROOT_DIR/preprocess/submodules/ScoreHMR/
    bash download_data.sh
    echo -e "\n"

    # reuse smpl model from sings data
    echo -e "\033[34m[${CURRENT_STAGE}] Copying [$ROOT_DIR/data/human_models/smpl/SMPL_NEUTRAL.pkl] to [$ROOT_DIR/preprocess/submodules/ScoreHMR/data/smpl/]...\033[0m"
    mkdir $ROOT_DIR/preprocess/submodules/ScoreHMR/data/smpl
    cp $ROOT_DIR/data/human_models/smpl/SMPL_NEUTRAL.pkl $ROOT_DIR/preprocess/submodules/ScoreHMR/data/smpl/
    echo -e "\n"

    # mv scorehmr script
    echo -e "\033[34m[${CURRENT_STAGE}] Copying [$ROOT_DIR/preprocess/utils/fit_for_sings.py] to [$ROOT_DIR/preprocess/submodules/ScoreHMR/]...\033[0m"
    cp $ROOT_DIR/preprocess/utils/fit_for_sings.py $ROOT_DIR/preprocess/submodules/ScoreHMR/
    echo -e "\n"
    
    # fix phalp smpl downloading url
    # conda activate $ENV_NAME
    cd $ROOT_DIR/preprocess/patches
    echo -e "\033[34m[${CURRENT_STAGE}] Copying [SMPL model] for [PHALP]...\033[0m"
    python fix_phalp_smpl.py
    echo -e "\033[32mDone.\033[0m"
    echo -e "\n"
}

prepare_alphapose
prepare_sam2
prepare_scorehmr

echo -e "\033[32m✅ All preparation steps completed successfully.  \033[0m"

# final test
bash quick_test_deps.sh