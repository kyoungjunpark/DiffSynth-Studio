#!/bin/bash

accelerate launch examples/wanvideo/model_training/train_with_ground_truth.py \
  --dataset_base_path data/coco_video_dataset,data/gptedit_video_dataset \
  --dataset_metadata_path data/coco_video_dataset/metadata.csv,data/gptedit_video_dataset/metadata_train_all.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_ground_truth_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32


# Example (recommended): use per-frame monotonic MSE + pseudo-target interpolation + monotonicity penalty
python -m accelerate.commands.launch examples/wanvideo/model_training/train_with_ground_truth.py \
  --dataset_base_path data/coco_video_dataset,data/gptedit_video_dataset \
  --dataset_metadata_path data/coco_video_dataset/metadata.csv,data/gptedit_video_dataset/metadata_train_all.csv \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --save_steps 10000 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_full_temporal_ma_mono" \
  --trainable_models "dit" \
  --use_per_frame_mse --temporal_frame_power 1.0 \
  --use_pseudo_target_interp --pseudo_interp_space latent \
  --use_monotonicity_loss --monotonicity_loss_weight 0.001 \
  --temporal_loss_weight 0.01 --temporal_normalize --temporal_warmup_steps 5000

accelerate launch examples/wanvideo/model_training/train_with_ground_truth.py \
  --dataset_base_path data/coco_video_dataset,data/gptedit_video_dataset \
  --dataset_metadata_path data/coco_video_dataset/metadata.csv,data/gptedit_video_dataset/metadata_train_all.csv \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --save_steps 10000 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_full_temporal_loss0_01_cap0_02_warmup5000" \
  --trainable_models "dit" \
  --temporal_loss_weight 0.01 \
  --temporal_cap 0.02 \
  --temporal_warmup_steps 5000

# + lpips in loss?

# simple version
accelerate launch examples/wanvideo/model_training/train_with_ground_truth.py \
  --dataset_base_path data/coco_video_dataset \
  --dataset_metadata_path data/coco_video_dataset/metadata.csv\
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --save_steps 1000 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_full_lastframe_only" \
  --trainable_models "dit" 

  python examples/wanvideo/model_training/validate_ground_truth_full.py \
  /blob/kyoungjun/DiffSynth-Studio/models/train/Wan2.2-TI2V-5B_full_temporal_ma_mono/step-40000.safetensors \
  data/coco_video_dataset/metadata_test.csv \
  ./validation_outputs_temporal_ma_mono_coco_40000steps \
  30
  
  # Give them a penalty if it is not moving at all.
