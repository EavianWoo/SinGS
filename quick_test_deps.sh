#!/bin/bash
# This script test the environment for the preprocessing.

set -euo pipefail
trap 'echo -e "\033[31m❌ [${CURRENT_STAGE}] failed at line $LINENO: $BASH_COMMAND\033[0m"' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "\033[31mStarting quick test of preprocessing tools...\033[0m"
# --------- Alphapose ----------
run_alphapose() {
    CURRENT_STAGE="AlphaPose"
    echo -e "\033[34m[${CURRENT_STAGE}] Starting...\033[0m"
    cd "$ROOT_DIR/preprocess/submodules/AlphaPose"

    python -m scripts.demo_inference \
        --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml \
        --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth \
        --video "$ROOT_DIR/examples/syn_videos/test_batch/f_1.mp4" \
        --outdir "$ROOT_DIR/examples/training_kits/test_batch/f_1/"

    echo -e "\033[32m[${CURRENT_STAGE}] Done.\033[0m"
}

# --------- SAM2 ----------
run_sam2() {
    CURRENT_STAGE="SAM2"
    echo -e "\033[34m[${CURRENT_STAGE}] Starting...\033[0m"
    cd "$ROOT_DIR/preprocess/submodules/sam2"

    python get_masks_for_sings.py \
        -v "$ROOT_DIR/examples/training_kits/test_batch/f_1/images" \
        -k "$ROOT_DIR/examples/training_kits/test_batch/f_1/alphapose-results.json" \
        -o mask_test

    echo -e "\033[32m[${CURRENT_STAGE}] Done.\033[0m"
}

# --------- ScoreHMR ----------
run_scorehmr() {
    CURRENT_STAGE="ScoreHMR"
    echo -e "\033[34m[${CURRENT_STAGE}] Starting...\033[0m"
    cd "$ROOT_DIR/preprocess/submodules/ScoreHMR"

    python demo_image.py \
    --img_folder example_data/images \
    --out_folder demo_out/images

    echo -e "\033[32m[${CURRENT_STAGE}] Done.\033[0m"
}

# --------- Execute all ----------
run_alphapose
run_sam2
run_scorehmr

echo -e "\033[32m✅ All modules finished successfully.\033[0m"
