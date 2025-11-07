#!/bin/bash
# This script is used to install sings environment and the dependencies for the preprocessing.
# We merge the enviroments to save space, you can install separately if needed.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_NAME="sings"


# ----------------------------------- #
# SinGS main requirements
if [[ "${1:-}" == "--main" ]]; then
    conda create -n $ENV_NAME python=3.10 -y
    conda activate $ENV_NAME

    pip install torch==2.5.1 torchvision==0.20.1 --extra-index-url https://download.pytorch.org/whl/cu121

    pip install -r requirements.txt

    pip install git+https://github.com/facebookresearch/pytorch3d.git
    pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
fi


# ----------------------------------- #
# SinGS custom dependencies
# If you need to process custom data, install the dependent environment.
if [[ "${1:-}" == "--deps" ]]; then
    # conda activate $ENV_NAME
    
    git submodule update --init --recursive

    ## ScoreHMR
    cd "$SCRIPT_DIR/preprocess/submodules/ScoreHMR"
    pip install phalp[all]@git+https://github.com/brjathu/PHALP.git

    pip install -U openmim
    mim install mmcv==1.5.0

    pip install -r requirements.txt
    git submodule update --init ViTPose
    pip install -v -e ViTPose


    ## AlphaPose
    pip install cython
    pip install numpy==1.23.5
    cd "$SCRIPT_DIR/preprocess/submodules/AlphaPose"
    python setup.py build_ext --inplace


    ## SAM2
    cd "$SCRIPT_DIR/preprocess/submodules/sam2"
    pip install -e . --no-deps
    python setup.py  build_ext --inplace


    # ## Sapiens (Optional)


fi
