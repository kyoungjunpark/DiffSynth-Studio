#!/bin/bash

BASE_MODEL_DIR="models/train/Wan2.2-TI2V-5B_full_temporal_loss0_01_norm_cap0_1_warmup5000"
SCRIPT="examples/wanvideo/model_training/validate_ground_truth_full.py"
NUM_SAMPLES=30

# COCO dataset
for STEP in 10000 20000 30000 50000; do
  python $SCRIPT \
    ${BASE_MODEL_DIR}/step-${STEP}.safetensors \
    data/coco_video_dataset/metadata_test.csv \
    ./validation_outputs_coco_temporal_loss0_01_norm_cap0_1_warmup5000_${STEP}steps \
    $NUM_SAMPLES
done

# GPT-Edit dataset
for STEP in 10000 20000 30000 50000; do
  python $SCRIPT \
    ${BASE_MODEL_DIR}/step-${STEP}.safetensors \
    data/gptedit_video_dataset/metadata_test_all.csv \
    ./validation_outputs_gptedit_temporal_loss0_01_norm_cap0_1_warmup5000_${STEP}steps \
    $NUM_SAMPLES
done

