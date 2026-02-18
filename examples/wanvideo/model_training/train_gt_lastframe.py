import torch, os
from diffsynth.pipelines.wan_video_gt_lastframe import WanVideoGTLastFramePipeline
from diffsynth.pipelines.wan_video_new import ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset, LoadImage, ImageCropAndResize, ToAbsolutePath
from examples.wanvideo.config.ground_truth_prompts import SYSTEM_PROMPT
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanGTLastFrameTrainingModule(DiffusionTrainingModule):
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
    ):
        super().__init__()
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = WanVideoGTLastFramePipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        
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
        self.first_step_printed = False
        
    def forward_preprocess(self, data):
        prompt = data["prompt"]
        if prompt.endswith(", "):
            prompt = prompt[:-2]
        full_prompt = SYSTEM_PROMPT + prompt
        
        inputs_posi = {"prompt": full_prompt}
        inputs_nega = {}
        
        # Use only reference image (like standard I2V)
        ref_img = data["reference_image"]
        gt_img = data["ground_truth_image"]
        
        inputs_shared = {
            "input_image": ref_img,  # Reference image for I2V conditioning
            "gt_image": gt_img,  # Ground truth for last frame supervision
            "height": ref_img.size[1],
            "width": ref_img.size[0],
            "num_frames": self.num_frames,
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # Pipeline units process: noise, prompt embedding, image conditioning
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    def forward(self, data, inputs=None):
        if inputs is None: 
            inputs = self.forward_preprocess(data)
        
        if not self.first_step_printed:
            print("\n=== First Training Sample ===")
            print(f"Reference Image: {data.get('reference_image_path', 'N/A')}")
            print(f"Ground Truth Image: {data.get('ground_truth_image_path', 'N/A')}")
            print(f"Original Prompt: {data.get('prompt', 'N/A')}")
            print(f"Full Prompt: {inputs['prompt']}")
            print("=" * 50 + "\n")
            self.first_step_printed = True
        
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        
        # Use the pipeline's training_loss method which handles GT last frame supervision
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    
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
    
    model = WanGTLastFrameTrainingModule(
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
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )

    launch_training_task(dataset, model, model_logger, args=args)
