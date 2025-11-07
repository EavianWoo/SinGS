<img src="./assets/dynamic_logo.gif" alt="Animation Result" style="width: 75%; height: auto;" />


# 🔥 News

[11-07-2025] Revisited and generalized the pipeline to support more generation models. 💃 Now you can try to distill **Wan2.2-Animate**.

[11-07-2025] Released the *reconstruction module* of **SinGS**.

<br><br>



# Paper & Page Links

**SinGS** creates high-quality, efficient, animatable avatar from just single image input. And this repository provides the official implementation for our paper [ *SinGS: Animatable Single-Image Human Gaussian Splats with Kinematic Priors* ] accepted by **CVPR 2025**.

[![arXiv](https://img.shields.io/badge/arXiv-2406.12345-b31b1b.svg)](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_SinGS_Animatable_Single-Image_Human_Gaussian_Splats_with_Kinematic_Priors_CVPR_2025_paper.pdf)	[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://yourdomain.com/project-page)	[![Video](https://img.shields.io/badge/Video-Demo-red)](https://github.com/EavianWoo/singsPage)

![Animation Result](./assets/LowAnim.gif)

<br><br>



# Installation

Our program has been test under the environments: CUDA ≥ 11.7 and Python ≥ 3.8.

- Clone the repository with

  ```
  git clone --recursive git@github.com:EavianWoo/SinGS.git
  ```


- Create a base environment

  ```
  source install_all.sh --main
  ```
  
  If you meet problems when installing pytorch3d, please visit the [[official guide]](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

<br><br>


# Preparation


- **Register on [SMPLify](https://smplify.is.tue.mpg.de) and [MANO](https://mano.is.tue.mpg.de)** and download the models with your account

  ```sh
  bash fetch_human_models.sh
  ```

- **[Optional] Test the main environment** and start a training with the provided data

  ```
  conda activate sings
  python scripts/train_avatar.py -c sings/rec/cfgs/train/beta/human_complex.yaml
  ```

- **Install environments and download data** for all dependencies

  (Only necessary if you plan to process custom data)

  ```
  conda activate sings
  bash install_all.sh --deps
  bash prepare_deps.sh
  ```
  You will see a message like "All modules finished successfully!" if everything completes successfully.
  Check the scripts 'prepare_deps.sh' if you need to download data manually.
  
<br><br>


# Quick Start

## Part 1: Generation

This part takes a single image and generates consecutive turn-around video.

> To be updated.

For now, you can use any generation model to replace this part.

If you'd like to distill **Wan2.2-Animate**, check out [[Distill Any]](./playground/doc/Custom.md#distill-any).<br><br>


## Part 2: Reconstruction & Animation

### 2.0 Data Preprocess

All input data for the project is organized under the `examples/` directory.

- Place your videos in `examples/syn_videos/`. You can also group them into subfolders for batch processing:

  ```text
  examples/
  ├── inputs
  ├── syn_videos/
  │		├── case_0.mp4
  │		└── batch_dir
  │				├── case_1.mp4
  │				├── case_2.mp4
  │				├── ...
  └── training_kits
  
  ```
  
  


- Run the scripts to preprocess videos

  **Single case**
  
  ```
  bash scripts/prepare_kits.sh examples/syn_videos/case_0.mp4
  ```
  
  **Batch**
  
  ```shell
  bash scripts/run_batch/prepare_kits_batch.sh examples/syn_videos/batch_dir
  ```



- After preprocessing, the processed data will be stored in `examples/training_kits/`.
   The directory structure should look like this:
  
  ```text
  examples/
  ├── inputs
  ├── syn_videos
  └── training_kits/
  		├── case_0
  		└── batch_dir
					├── case_1
					├── case_2
					├── ...
  
  ```

Please see the [[Custom Guide]](./playground/doc/Custom.md) for more details.



### 2.1 Train

**Once Training**

- Train an avatar using a specific config file

  ```shell
  python scripts/train_avatar.py -c sings/rec/cfgs/train/beta/human_male.yaml
  ```

  You can also override specific settings from the command line, for example:

  ```shell
  python scripts/train_avatar.py -c <path/to/config> dataset.batch='batch_dir' dataset.name='case_1'
  ```



**Batch Training**

- To train multiple avatars in a batch, organize your cases into a batch directory and run

  ```
  bash scripts/run_batch/train_batch.sh examples/training_kits/batch_dir [path/to/config]
  ```

  

### 2.2 Animate

- Animate the avatar from a designated result directory.

  ```shell
  python scripts/anim_avatar.py -o <path/to/case_out_dir>
  ```

  Set  `anim_cfg_path` if you need to change the motion sequence.

  For animation, two types of input signals are supported:

  1. Public motion datasets (e.g., AMASS).

  2. Custom motion sequences extracted from videos.
  
  > **NOTE**
  >
  > We provide several example pose sequences from **AMASS** for quick validation. If you need more motion sequences please visit [[AMASS]](https://amass.is.tue.mpg.de/download.php);
  >
  > If you are interested in extracting your own motion sequence, please refer to [[Extract Motion]](./playground/doc/Motion.md).

<br><br>



# Upcoming Releases

The following components will be open-sourced soon:

- [x] The implemention of Geometry-Preserving 3DGS.

- [x] The complete preprocess code for custom data.

- [ ] The generation model and  pretrained weights.

<br><br>



# Acknowledgements

Parts of the code are adapted from the following repositories:

- [HUGS](https://github.com/apple/ml-hugs)
- [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
- [4DGS](https://github.com/hustvl/4DGaussians)

We also make use of the following excellent open-source projects:

- [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
- [SAM2](https://github.com/facebookresearch/segment-anything-2)
- [ScoreHMR](https://github.com/statho/ScoreHMR)
- [Sapiens](https://github.com/facebookresearch/sapiens)

We sincerely thank all the authors and contributors of these projects.
