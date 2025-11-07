### Environtment Setup

First, install the required environment and download necessary data for preprocessing.

```shell
source install_all.sh --main
source prepare_all.sh --deps
```

The data preparation now involves the following steps:

- **Keypoint Detection**: AlphaPose
- **Segmentation**: Sam2
- **SMPL Fitting**: ScoreHMR.
- **SMPL optimization (Optional)** : Sapiens

<br><br>


### Generation

After the environment set down, you can generate and process custom images.

- **Organize your images**  in the folder `examples/inputs/`

  ```text
  examples/
  └── inputs
      └── your_custom_idr
          ├── img_0.png
          └── img_2.jpg
  
  ```
  
  


- Run the generation script

  ```
  python scripts/run_batch/infer_videos_batch.sh examples/inputs/your_custom_dir
  ```

  The video results will be saved in `examples/syn_videos/your_custom_dir`.



- Prepare training kit

  ```
  source scripts/run_batch/prepare_kits_batch.sh examples/syn_videos/test_batch [--opt]
  ```

  The training data will be saved in `examples/syn_videos/your_custom_dir`.

  > NOTE
  >
  > - The `--opt` flag enables SMPL optimization using **Sapiens** keypoints.
  > - SMPL optimization requires the SAPIENS environment and may take several additional minutes per batch.

<br><br>


### Distill Any

You can use this module to distill other generation models. 
For example, to distill **Wan2.2-Animate**:

- Use your image as the reference and our provided video (e.g., examples/syn_videos/test_batch/m_1.mp4) as target for generation.

  > ...

  

- Put the generated video in  `examples/syn_videos/` (directly or organize them in folder)

    ```text
    examples/
    ├── inputs
    ├── syn_videos/
    │		├── case_0.mp4
    │		└── batch_dir
    │				├── case_1.mp4
    │				├── case_2.mp4
    │       ├── ...
    └── training_kits
    
    ```

  

- Run the preprocessing script to prepare the data (once or batch)

  ```shell
  bash scripts/prepare_kits.sh examples/syn_videos/case_0.mp4 [--opt]
  # or
  bash scripts/run_batch/prepare_kits_batch.sh examples/syn_videos/test_batch [--opt]
  ```
  Check `examples/training_kits/` after processing.

  

- Train the avatar (once or batch)

  ```shell
  python scripts/train_avatar.py -c <path/to/config> dataset.name='case_0'
  # or
  bash scripts/run_batch/train_batch.sh examples/training_kits/batch_dir [path/to/config]
  ```
  The default configs are store in `sings/rec/cfgs/beta/`



- Animate the avatar

  ```shell
  python scripts/anim_avatar.py -o <path/to/case_out_dir>
  ```

  
