#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

BASE_MODEL_DIR="models/train/Wan2.2-TI2V-5B_full_lastframe_only_final/"
BASE_MODEL_DIR="models/train/Wan2.2-TI2V-5B_full_lastframe_only"
SCRIPT="examples/wanvideo/model_training/validate_ground_truth_full.py"
NUM_SAMPLES=30

# 마지막 폴더명 추출
MODEL_NAME=$(basename "$BASE_MODEL_DIR")

echo "Finding checkpoints in $BASE_MODEL_DIR"

# Find all step checkpoints and sort them numerically
# CHECKPOINTS=$(find "$BASE_MODEL_DIR" -name "step-10000.safetensors")
CHECKPOINTS=$(find "$BASE_MODEL_DIR" -name "step-*.safetensors")
CHECKPOINTS=$(find "$BASE_MODEL_DIR" -name "step-*.safetensors" | sort -V)

if [ -z "$CHECKPOINTS" ]; then
    echo "No checkpoints found in $BASE_MODEL_DIR"
    exit 1
fi

echo "Found checkpoints:"
echo "$CHECKPOINTS"
echo ""

# Loop through each checkpoint - alternate between COCO and GPT-Edit
for CKPT_PATH in $CHECKPOINTS; do
    # Extract step number from filename
    STEP=$(basename "$CKPT_PATH" | sed 's/step-\([0-9]*\)\.safetensors/\1/')
    
    # COCO dataset
    OUTPUT_DIR="./validation_outputs_coco_${MODEL_NAME}_${STEP}steps"
    
    echo "========================================="
    echo "Running COCO validation for step-${STEP}"
    echo "Checkpoint: $CKPT_PATH"
    echo "Output dir: $OUTPUT_DIR"
    echo "========================================="
    
    python $SCRIPT \
        "$CKPT_PATH" \
        data/coco_video_dataset/metadata_val_final.csv \
        "$OUTPUT_DIR" \
        $NUM_SAMPLES
    
    if [ $? -eq 0 ]; then
        echo "✓ COCO validation completed for step-${STEP}"
    else
        echo "✗ COCO validation failed for step-${STEP}"
    fi
    echo ""
    
    # GPT-Edit dataset
    OUTPUT_DIR="./validation_outputs_gptedit_${MODEL_NAME}_${STEP}steps"
    
    echo "========================================="
    echo "Running GPT-Edit validation for step-${STEP}"
    echo "Checkpoint: $CKPT_PATH"
    echo "Output dir: $OUTPUT_DIR"
    echo "========================================="
    
    python $SCRIPT \
        "$CKPT_PATH" \
        data/gptedit_video_dataset/metadata_test_all.csv \
        "$OUTPUT_DIR" \
        $NUM_SAMPLES
    
    if [ $? -eq 0 ]; then
        echo "✓ GPT-Edit validation completed for step-${STEP}"
    else
        echo "✗ GPT-Edit validation failed for step-${STEP}"
    fi
    echo ""
done

echo "All validations completed!"

