#!/bin/bash

export CUDA_VISIBLE_DEVICES=3


BASE_MODEL_DIR="models/train/Wan2.2-TI2V-5B_full_temporal_loss0_01_cap0_02_warmup5000/"
# BASE_MODEL_DIR="models/train/Wan2.2-TI2V-5B_full_temporal_loss0_01_norm_cap0_1_warmup5000/"
# BASE_MODEL_DIR="models/train/Wan2.2-TI2V-5B_full_temporal_ma_mono/"
SCRIPT="examples/wanvideo/model_training/validate_ground_truth_full.py"
NUM_SAMPLES=30

# 마지막 폴더명 추출
MODEL_NAME=$(basename "$BASE_MODEL_DIR")

# COCO dataset
for STEP in 80000 90000; do
  python $SCRIPT \
    ${BASE_MODEL_DIR}/step-${STEP}.safetensors \
    data/coco_video_dataset/metadata_val_final.csv \
    ./validation_outputs_coco_${MODEL_NAME}_${STEP}steps \
    $NUM_SAMPLES
done

# GPT-Edit dataset
for STEP in 80000 90000; do
  python $SCRIPT \
    ${BASE_MODEL_DIR}/step-${STEP}.safetensors \
    data/gptedit_video_dataset/metadata_test_all.csv \
    ./validation_outputs_gptedit_${MODEL_NAME}_${STEP}steps \
    $NUM_SAMPLES
done

