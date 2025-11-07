#!bin/bash

### cmd ###
# bash scripts/prepare_kits.sh example/syn_videos/women.mp4
VIDEO_PATH=$1

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR=$(realpath $(dirname "$CURRENT_DIR"))
WORK_DIR="$PROJ_DIR/examples/training_kits"

# Parse command line arguments
OPTIMIZE=false
for arg in "$@"
do
    if [ "$arg" == "--opt" ]; then
        OPTIMIZE=true
    fi
done

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


# conda activate sings

# Step 1 extract frames
cd preprocess/utils
python extract_frames.py -v ${VIDEO_PATH} -o ${KIT_DIR}


# Step 2 get masks
cd $PROJ_DIR/preprocess/submodules/AlphaPose
python -m scripts.demo_inference \
--cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml \
--checkpoint pretrained_models/halpe26_fast_res50_256x192.pth \
--video ${VIDEO_PATH} \
--outdir ${KIT_DIR} \
# --save_img \
# --showbox

# check
if [ ! -d "$IMAGE_DIR" ]; then
    echo "images directory: $IMAGE_DIR does not exist!"
fi
if [ ! -e "$KEYPOINT_PATH" ]; then
    echo "keypoint file: $KEYPOINT_PATH does not exist!"
fi

cd $PROJ_DIR/preprocess/submodules/sam2
python get_masks_for_sings.py -v ${IMAGE_DIR} -k ${KEYPOINT_PATH} --use_box

cd ${MASK_DIR}
ffmpeg -framerate 30 -i %06d.png -c:v libx264 -r 30 -pix_fmt yuv420p output.mp4

# Step 3 - get smpl_fits
cd $PROJ_DIR/preprocess/submodules/ScoreHMR
python fit_for_sings.py --input_path ${KIT_DIR} --out_folder ${KIT_DIR}


# # # refine pose