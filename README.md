![](./assets/dynamic_logs.gif)

# Paper & Page Links

**SinGS** creates high-quality, efficient, animatable avatar from just single image input. And this repository provides the official implementation for our paper [ *SinGS: Animatable Single-Image Human Gaussian Splats with Kinematic Priors* ]** accepted by CVPR 2025.

[![arXiv](https://img.shields.io/badge/arXiv-2406.12345-b31b1b.svg)](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_SinGS_Animatable_Single-Image_Human_Gaussian_Splats_with_Kinematic_Priors_CVPR_2025_paper.pdf)	[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://github.com/EavianWoo/singsPage)	[![Video](https://img.shields.io/badge/Video-Demo-red)](https://www.youtube.com/watch?v=2NVqoVNVmjY&t=13s)

![Animation Result](./assets/FinalAnim.mp4)



# Installation

Our program has been test under the environments CUDA ≥ 11.7 and Python ≥ 3.8 environments.

- Clone the repository with

  ```
  git clone --recursive git@github.com:EavianWoo/SinGS.git
  ```

  

- Create a base environment

  ```
  cd SinGS
  source quick_set.sh
  pip install -e .
  ```

  If you meet problems when installing pytorch3d, please visit the [official guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
  
  



# Preparation

- Register SMPL offcial website and download the SMPL model with your account

  ```sh
  bash fetch_human_models.sh
  ```

  



# Quick Start

### Part 1: Generation

This part takes a single image and generates consecutive turn-around video.

To be updates. Please check out the [custom guide](./custom.md)



### Part 2: Reconstruction & Animation

This part reconstruct animatable avatars which is easy to control. 

- train the avatar with the video sequce

  ```
  python script/train_avatar.py --cfg_file dataset.
  ```

  And it support to slightly adjust the arguments with cmd

  ```
  python script/train_avatar.py --cfg_file dataset.
  ```



- animate the avatar from a designated result file.

  ```
  python script/anim_avatar.py -out_dir
  ```

  For animation, we provide two kinds of input signals:

  1. Public motion datasets, e.g., AMASS.

  2. Custom motion sequences extracted from videos.

  If you are interested in extract your own motion sequence, 





# Upcoming Releases

The following components will be open-sourced soon:

- [x] The implemention of Geometry-Preserving 3DGS.
- [ ] The complete preprocess code for custom data.
- [ ] A generation model and  pretrained weights.
- [ ] A real-time motion mimic demo from the camera.
- [ ] A playground with some handy tools involving, like real-time video pose extraction, multi-character rendering in a single scene and more maybe~

