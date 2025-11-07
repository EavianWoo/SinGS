#!/bin/bash

### Command ###
# bash prepare_kits_batch_optimized_smpl.sh ../../examples/syn_videos/test_batch [--opt]
# append [-opt] if keypoint optimization is enabled

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR=$(dirname $(dirname "$CURRENT_DIR"))
echo "Project directory: $PROJ_DIR"

BATCH_DIR=$1 # NOTE directory of videos 
BATCH_DIRNAME=$(basename "$BATCH_DIR")
WORK_DIR="$PROJ_DIR/examples/training_kits/$BATCH_DIRNAME"

# Whether to use keypoint optimization
OPTIMIZE=false
for arg in "$@"
do
    if [ "$arg" == "--opt" ]; then
        OPTIMIZE=true
    fi
done


if [ -z "$1" ]; then
    echo "Please provide the batch folder"
    exit 1
fi

if [ ! -d "$WORK_DIR" ]; then
    mkdir -p "$WORK_DIR"
    echo "Make directory: $WORK_DIR"
fi


# Count videos
VIDEO_COUNT=$(ls "$BATCH_DIR"/*.mp4 2>/dev/null | wc -l)
if [ "$VIDEO_COUNT" -eq 0 ]; then
  echo "No video files found in [$BATCH_DIR]"
  exit 1
else
  echo "$VIDEO_COUNT video(s) have been found in [$BATCH_DIR]"
fi


for VIDEO_PATH in ${BATCH_DIR}/*.mp4; do

    VIDEO_PATH=$(realpath "$VIDEO_PATH")
    VIDEO_NAME=$(basename "$VIDEO_PATH" | cut -f 1 -d '.')
    
    KIT_DIR="$WORK_DIR/$VIDEO_NAME"
    IMAGE_DIR="$KIT_DIR/images"
    MASK_DIR="$KIT_DIR/masks"
    KEYPOINT_PATH="$KIT_DIR/alphapose-results.json"

    if [ ! -d "$KIT_DIR" ]; then
        mkdir -p "$KIT_DIR"
        echo "Make directory: $KIT_DIR"
    fi

    if [ ! -d "$IMAGE_DIR" ]; then
        mkdir -p "$IMAGE_DIR"
        echo "Make directory: $IMAGE_DIR"
    fi

    echo "Start processing video [${VIDEO_NAME}]..."

    # Step 1 extract frames
    # conda activate sings
    echo "[Step 1] Extracting frames from ${VIDEO_PATH}"
    cd "$PROJ_DIR/preprocess/utils"
    python extract_frames.py -v ${VIDEO_PATH} -o ${KIT_DIR}


    # Step 2 get masks
    # conda activate alphapose # if separate env
    echo "[Step 2.1] Detecting poses on ${VIDEO_PATH}"
    cd "$PROJ_DIR/preprocess/submodules/AlphaPose"
    python -m scripts.demo_inference \
    --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml \
    --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth \
    --video ${VIDEO_PATH} \
    --outdir ${KIT_DIR} \
    # --save_img \
    # --showbox

    # check files
    echo "Check prerequisites for segmentation..."
    if [ ! -d "$IMAGE_DIR" ]; then
        echo "images directory: [$IMAGE_DIR] does not exist!"
        exit 1
    fi
    if [ ! -e "$KEYPOINT_PATH" ]; then
        echo "keypoint file: [$KEYPOINT_PATH] does not exist!"
        exit 1
    fi

    # conda activate sam2 # if separate env
    echo "[Step 2.2] Generating masks for ${VIDEO_PATH}"
    cd "$PROJ_DIR/preprocess/submodules/sam2"
    python get_masks_for_sings.py -v ${IMAGE_DIR} -k ${KEYPOINT_PATH} --use_box

    cd ${MASK_DIR}
    ffmpeg -framerate 30 -i %06d.png -c:v libx264 -r 30 -pix_fmt yuv420p output.mp4


    # Step 3 - get smpl fits
    # conda activate ScoreHMR # if separate env
    echo "[Step 3] Getting SMPL fits for ${VIDEO_PATH}"
    cd "$PROJ_DIR/preprocess/submodules/ScoreHMR"
    python fit_for_sings.py --input_path ${KIT_DIR} --out_folder ${KIT_DIR}


    # # Step 4 - refine pose
    # # (Optional)
    # if [ "$OPTIMIZE" = true ]; then
    #     KP_DIR="$KIT_DIR/keypoints_coco133"
    #     # conda activate sapiens # if separate env
    #     echo "[Step 4.1] Detecting keypoints for ${VIDEO_PATH}"
    #     cd "$PROJ_DIR/preprocess/submodules/sapiens/pose/scripts/demo/local"
    #     bash ./keypoints133_edited.sh ${IMAGE_DIR} ${KP_DIR}

    #     conda activate sings
    #     echo "[Step 4.2] Optimizing SMPL fits for ${VIDEO_PATH}"
    #     cd "$PROJ_DIR/preprocess/utils"
    #     python ooptimize_smplh.py --kit_dir ${KIT_DIR}
    # else
    #     echo "[Step 4] Skipping SMPL optimization step."
    # fi

done