import torch, os
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset, LoadImage, ImageCropAndResize, ToAbsolutePath
from examples.wanvideo.config.ground_truth_prompts import SYSTEM_PROMPT
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanGroundTruthTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        num_frames=49,
        temporal_loss_weight=0.0,
        temporal_normalize=False,
        temporal_cap=0.0,
        temporal_warmup_steps=0,
        # New options
        use_per_frame_mse=False,
        temporal_frame_power=1.0,
        use_pseudo_target_interp=False,
        pseudo_interp_space='latent',
        use_monotonicity_loss=False,
        monotonicity_loss_weight=0.0,
    ):
        super().__init__()
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.num_frames = num_frames
        self.temporal_loss_weight = temporal_loss_weight
        self.temporal_normalize = temporal_normalize
        self.temporal_cap = temporal_cap
        self.temporal_warmup_steps = temporal_warmup_steps
        # New attributes
        self.use_per_frame_mse = use_per_frame_mse
        self.temporal_frame_power = temporal_frame_power
        self.use_pseudo_target_interp = use_pseudo_target_interp
        self.pseudo_interp_space = pseudo_interp_space
        self.use_monotonicity_loss = use_monotonicity_loss
        self.monotonicity_loss_weight = monotonicity_loss_weight
        self.first_step_printed = False
        
    def forward_preprocess(self, data):
        prompt = data["prompt"]
        if prompt.endswith(", "):
            prompt = prompt[:-2]
        full_prompt = SYSTEM_PROMPT + prompt
        
        inputs_posi = {"prompt": full_prompt}
        inputs_nega = {}
        
        # Create video with reference at start and ground_truth at end
        ref_img = data["reference_image"]
        gt_img = data["ground_truth_image"]
        video_frames = [ref_img] + [gt_img] * (self.num_frames - 1)
        
        inputs_shared = {
            "input_image": ref_img,
            "input_video": video_frames,
            "height": ref_img.size[1],
            "width": ref_img.size[0],
            "num_frames": self.num_frames,
            "training_step": data.get("training_step", 0),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        
        if not self.first_step_printed:
            print("\n=== First Training Sample ===")
            print(f"Reference Image: {data.get('reference_image_path', 'N/A')}")
            print(f"Ground Truth Image: {data.get('ground_truth_image_path', 'N/A')}")
            print(f"Original Prompt: {data.get('prompt', 'N/A')}")
            print(f"Full Prompt: {inputs['prompt']}")
            print("=" * 50 + "\n")
            self.first_step_printed = True
        
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        
        # Calculate loss only on the last frame
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.pipe.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.pipe.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        
        inputs["latents"] = self.pipe.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.pipe.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        
        noise_pred = self.pipe.model_fn(**models, **inputs, timestep=timestep)
        
        # Optionally perform pseudo-target interpolation in latent space for intermediate frames
        training_target_for_loss = training_target
        if self.use_pseudo_target_interp:
            if self.pseudo_interp_space != 'latent':
                # Image-space interpolation is expensive; warn and fall back to latent
                print("Warning: pseudo-target interpolation in image space is not implemented; falling back to latent-space interpolation.")
            # training_target shape: B, C, T, H, W
            B, C, T, H, W = training_target.shape
            first = training_target[:, :, 0:1, :, :]
            last = training_target[:, :, -1:, :, :]
            # alpha values from 0..1 for each frame
            alpha = torch.linspace(0.0, 1.0, steps=T, device=training_target.device, dtype=training_target.dtype).view(1, 1, T, 1, 1)
            training_target_for_loss = first * (1.0 - alpha) + last * alpha

        # Extract only the last frame from latents (shape: B, C, T, H, W)
        # Last frame index is -1 in the T dimension
        noise_pred_last = noise_pred[:, :, -1:, :, :]
        training_target_last = training_target_for_loss[:, :, -1:, :, :]

        last_mse = torch.nn.functional.mse_loss(noise_pred_last.float(), training_target_last.float())
        training_weight = self.pipe.scheduler.training_weight(timestep)

        # Compute per-frame MSEs (averaged over batch, channel, spatial dims)
        sq = (noise_pred.float() - training_target_for_loss.float()) ** 2
        # shape: B, C, T, H, W -> mean over B,C,H,W -> T
        mse_per_frame = sq.mean(dim=(0, 1, 3, 4))

        # Per-frame monotonic weighting
        if self.use_per_frame_mse:
            T = mse_per_frame.shape[0]
            if T > 1:
                positions = torch.arange(0, T, device=mse_per_frame.device, dtype=mse_per_frame.dtype)
                denom = float(T - 1)
                weights = (positions / denom) ** float(self.temporal_frame_power)
            else:
                weights = torch.tensor([1.0], device=mse_per_frame.device, dtype=mse_per_frame.dtype)
            # normalize weights so total scale is similar to mean
            weights = weights / (weights.sum() + 1e-12)
            total_mse = (mse_per_frame * weights).sum()
            loss_data_term = total_mse
        else:
            loss_data_term = last_mse

        # Monotonicity loss on distance-to-GT
        monotonicity_penalty = torch.tensor(0.0, device=training_target.device)
        if self.use_monotonicity_loss:
            diffs = mse_per_frame[1:] - mse_per_frame[:-1]
            relu = torch.clamp(diffs, min=0.0)
            monotonicity_penalty = relu.sum()

        if self.temporal_loss_weight and self.temporal_loss_weight != 0.0:
            # Encourage frame-to-frame change by rewarding temporal differences in predictions.
            # temporal_change is mean squared diff between consecutive frames across T dimension.
            pred_diffs = noise_pred[:, :, 1:, :, :] - noise_pred[:, :, :-1, :, :]
            temporal_change = torch.mean(pred_diffs.float() * pred_diffs.float())

            # Optional normalization by global prediction magnitude to reduce scale sensitivity.
            if self.temporal_normalize:
                denom = torch.mean(noise_pred.float() * noise_pred.float()) + 1e-8
                temporal_change = temporal_change / denom

            # Optional cap to limit maximum reward contribution.
            if self.temporal_cap and self.temporal_cap > 0.0:
                temporal_change = torch.clamp(temporal_change, max=self.temporal_cap)

            # Warmup schedule based on training step (injected into inputs)
            training_step = int(inputs.get("training_step", 0))
            if self.temporal_warmup_steps and self.temporal_warmup_steps > 0:
                scale = min(1.0, float(training_step) / float(self.temporal_warmup_steps))
            else:
                scale = 1.0
            effective_weight = self.temporal_loss_weight * scale

            # Combine terms: data-term, temporal reward (subtracted), and monotonicity penalty (added)
            loss = loss_data_term * training_weight - effective_weight * temporal_change * training_weight + self.monotonicity_loss_weight * monotonicity_penalty * training_weight
        else:
            loss = loss_data_term * training_weight + self.monotonicity_loss_weight * monotonicity_penalty * training_weight

        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

    # Allow quick testing by specifying TEST_SAVE_STEP env var (overrides missing CLI arg)
    if not hasattr(args, 'test_save_step') or getattr(args, 'test_save_step') is None:
        env_val = os.environ.get('TEST_SAVE_STEP')
        if env_val is not None:
            try:
                args.test_save_step = int(env_val)
            except Exception:
                print('Warning: TEST_SAVE_STEP env var is not a valid integer; ignoring')
    
    def _split_csv(value):
        if value is None:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]
    
    class PathPreservingDataset(UnifiedDataset):
        def __getitem__(self, data_id):
            result = super().__getitem__(data_id)
            if result is not None:
                original_data = self.data[data_id % len(self.data)]
                result['reference_image_path'] = original_data.get('reference_image', 'N/A')
                result['ground_truth_image_path'] = original_data.get('ground_truth_image', 'N/A')
            return result
    
    base_paths = _split_csv(args.dataset_base_path)
    metadata_paths = _split_csv(args.dataset_metadata_path)
    if not base_paths:
        raise ValueError("dataset_base_path is required.")
    if not metadata_paths:
        raise ValueError("dataset_metadata_path is required.")
    if len(base_paths) != len(metadata_paths):
        raise ValueError(
            "dataset_base_path and dataset_metadata_path must have the same number of entries."
        )
    
    datasets = []
    for base_path, metadata_path in zip(base_paths, metadata_paths):
        datasets.append(
            PathPreservingDataset(
                base_path=base_path,
                metadata_path=metadata_path,
                repeat=args.dataset_repeat,
                data_file_keys=["reference_image", "ground_truth_image"],
                main_data_operator=ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(
                    args.height, args.width, args.max_pixels, 16, 16
                ),
            )
        )
    
    dataset = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
    
    model = WanGroundTruthTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        num_frames=args.num_frames,
        temporal_loss_weight=getattr(args, 'temporal_loss_weight', 0.0),
        temporal_normalize=getattr(args, 'temporal_normalize', False),
        temporal_cap=getattr(args, 'temporal_cap', 0.0),
        temporal_warmup_steps=getattr(args, 'temporal_warmup_steps', 0),
        # New options
        use_per_frame_mse=getattr(args, 'use_per_frame_mse', False),
        temporal_frame_power=getattr(args, 'temporal_frame_power', 1.0),
        use_pseudo_target_interp=getattr(args, 'use_pseudo_target_interp', False),
        pseudo_interp_space=getattr(args, 'pseudo_interp_space', 'latent'),
        use_monotonicity_loss=getattr(args, 'use_monotonicity_loss', False),
        monotonicity_loss_weight=getattr(args, 'monotonicity_loss_weight', 0.0),
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        test_save_step=getattr(args, 'test_save_step', None)
    )

    launch_training_task(dataset, model, model_logger, args=args)
