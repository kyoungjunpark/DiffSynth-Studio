"""
WanVideo Pipeline for Ground Truth Last Frame Training
Extends WanVideoPipeline to support training with GT supervision on the last frame only.
"""

import torch
from .wan_video_new import WanVideoPipeline


class WanVideoGTLastFramePipeline(WanVideoPipeline):
    """
    Pipeline for training with ground truth last frame supervision.
    
    Usage:
        - Set input_image: reference image for I2V conditioning
        - Set gt_image: ground truth image for last frame supervision
        - Training loss computed only on last frame
    """
    
    def training_loss_gt_lastframe(self, **inputs):
        """
        Custom training loss for GT last frame supervision.
        
        Args:
            inputs: Should contain:
                - latents: noised latents (B, C, T, H, W)
                - noise: noise tensor
                - gt_image: ground truth image for last frame (PIL Image or preprocessed)
                - Other model inputs (context, timestep, etc.)
        
        Returns:
            loss: MSE loss on last frame only
        """
        # Sample timestep
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        
        # Add noise to latents
        inputs["latents"] = self.scheduler.add_noise(inputs["latents"], inputs["noise"], timestep)
        
        # Standard training target from noised latents
        training_target = self.scheduler.training_target(inputs["latents"], inputs["noise"], timestep)
        
        # Encode GT image for last frame supervision
        gt_image = inputs.pop("gt_image", None)
        if gt_image is not None:
            self.load_models_to_device(["vae"])
            
            # Preprocess and encode GT image
            if not isinstance(gt_image, torch.Tensor):
                gt_video = self.preprocess_video([gt_image])
            else:
                gt_video = gt_image
                
            gt_latents = self.vae.encode(
                gt_video,
                device=self.device,
                tiled=inputs.get("tiled", False),
                tile_size=inputs.get("tile_size", None),
                tile_stride=inputs.get("tile_stride", None),
            ).to(dtype=self.torch_dtype, device=self.device)
            gt_latent = gt_latents[:, :, 0:1, :, :]  # Shape: [B, C, 1, H, W]
            
            # Use same noise slice for GT as last frame for consistency
            gt_noise = inputs["noise"][:, :, -1:, :, :]
            gt_training_target = self.scheduler.training_target(gt_latent, gt_noise, timestep)
            
            # Replace only the last frame in training_target
            training_target = training_target.clone()
            training_target[:, :, -1:, :, :] = gt_training_target
        
        # Forward pass through the model
        noise_pred = self.model_fn(**inputs, timestep=timestep)
        
        # Compute loss only on the last frame
        noise_pred_last = noise_pred[:, :, -1:, :, :]
        training_target_last = training_target[:, :, -1:, :, :]
        
        loss = torch.nn.functional.mse_loss(noise_pred_last.float(), training_target_last.float())
        loss = loss * self.scheduler.training_weight(timestep)
        
        return loss
    
    def training_loss(self, **inputs):
        """
        Override training_loss to use GT last frame supervision if gt_image is provided.
        Falls back to standard training_loss if gt_image is not provided.
        """
        if "gt_image" in inputs and inputs["gt_image"] is not None:
            return self.training_loss_gt_lastframe(**inputs)
        else:
            return super().training_loss(**inputs)
