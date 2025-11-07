#!/bin/bash

### Command ###
# bash train_batch.sh ../../examples/training_kits/test_batch [cfg_file]
# if cfg_file is not provided, the default cfg file will be used

CURRENT_DIR=$(realpath $(dirname "$0"))
PROJ_DIR=$(dirname $(dirname "$CURRENT_DIR"))
cd ${PROJ_DIR}


BATCH_DIR=$1
BATCH_DIRNAME=$(basename "$BATCH_DIR")

if [ -n "$2" ]; then
    CFG_FILE=$2
else
    CFG_FILE="$PROJ_DIR/sings/rec/cfgs/train/beta/human_male.yaml"
fi


echo "Batch directory: $BATCH_DIR"
KIT_COUNT=$(ls "$BATCH_DIR" 2>/dev/null | wc -l)
if [ "$KIT_COUNT" -eq 0 ]; then
    echo "No tranining kit found in $BATCH_DIR"
    exit 1
else
    echo "$KIT_COUNT training kit(s) have been found in folder [$BATCH_DIR]."
fi
echo "Batch process in this folder is about to begin..."


for KIT_PATH in $"$BATCH_DIR"/*; do
    KIT_NAME=$(basename "$KIT_PATH")
    echo "Runing experiment on dataset [${KIT_NAME}]"
    python scripts/train_avatar.py --cfg_file=$CFG_FILE \
    dataset.batch=${BATCH_DIRNAME} dataset.name=${KIT_NAME}
done

