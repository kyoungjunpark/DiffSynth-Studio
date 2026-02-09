import torch, os, sys, json
from datetime import datetime
from PIL import Image
import numpy as np
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import lpips
from examples.wanvideo.config.ground_truth_prompts import SYSTEM_PROMPT

def calculate_metrics(generated_frame, ground_truth_frame):
    # Convert PIL to numpy arrays
    gen_np = np.array(generated_frame).astype(np.float32) / 255.0
    gt_np = np.array(ground_truth_frame).astype(np.float32) / 255.0
    
    # MSE
    mse = np.mean((gen_np - gt_np) ** 2)
    
    # PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # SSIM
    ssim_value = ssim(gt_np, gen_np, channel_axis=2, data_range=1.0)
    
    # LPIPS
    lpips_model = lpips.LPIPS(net='alex').cuda()
    gen_tensor = torch.from_numpy(gen_np).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
    gt_tensor = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
    lpips_value = lpips_model(gen_tensor, gt_tensor).item()
    
    return {'mse': mse, 'psnr': psnr, 'ssim': ssim_value, 'lpips': lpips_value}

def validate(lora_checkpoint_path, test_metadata_path, output_dir, num_samples=5):
    os.makedirs(output_dir, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
        ],
    )
    pipe.load_lora(pipe.dit, lora_checkpoint_path, alpha=1)
    pipe.enable_vram_management()
    
    metadata = pd.read_csv(test_metadata_path)
    all_metrics = []
    
    for i in range(min(num_samples, len(metadata))):
        row = metadata.iloc[i]
        ref_img_path = row['reference_image']
        gt_img_path = row['ground_truth_image']
        prompt = row['prompt']
        if prompt.endswith(", "):
            prompt = prompt[:-2]
        full_prompt = SYSTEM_PROMPT + prompt
        
        input_image = Image.open(ref_img_path).convert("RGB")
        ground_truth_image = Image.open(gt_img_path).convert("RGB").resize((832, 480))
        
        video = pipe(
            prompt=full_prompt,
            negative_prompt="No camera or subject motion. No zoom, rotation, flicker, or lighting change. No background movement or distortion. No new objects, artifacts, shadows, reflections, or text. Bright colors, overexposed, static, blurry details, subtitles, style, artwork, image, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed limbs, fused fingers, still frame, messy background, three legs, crowded background people, walking backwards",
            input_image=input_image,
            num_frames=49,
            height=480,
            width=832,
            seed=0,
            tiled=True,
        )
        
        output_path = os.path.join(output_dir, f"sample_{run_tag}_{i}.mp4")
        ref_path = os.path.join(output_dir, f"sample_{run_tag}_{i}_reference.png")
        gt_path = os.path.join(output_dir, f"sample_{run_tag}_{i}_ground_truth.png")
        prompt_path = os.path.join(output_dir, f"sample_{run_tag}_{i}_prompt.txt")
        input_image.save(ref_path)
        ground_truth_image.save(gt_path)
        with open(prompt_path, "w") as f:
            f.write(full_prompt)
        save_video(video, output_path, fps=15, quality=5)
        
        # Calculate metrics on last frame
        last_frame = video[-1]
        metrics = calculate_metrics(last_frame, ground_truth_image)
        metrics['sample_id'] = i
        all_metrics.append(metrics)
        
        print(f"Sample {i+1}/{num_samples} - MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}, LPIPS: {metrics['lpips']:.4f}")
    
    # Calculate average metrics
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in all_metrics]),
        'psnr': np.mean([m['psnr'] for m in all_metrics]),
        'ssim': np.mean([m['ssim'] for m in all_metrics]),
        'lpips': np.mean([m['lpips'] for m in all_metrics])
    }
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump({'per_sample': all_metrics, 'average': avg_metrics}, f, indent=2)
    
    print(f"\n=== Average Metrics ===")
    print(f"MSE: {avg_metrics['mse']:.6f}")
    print(f"PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"SSIM: {avg_metrics['ssim']:.4f}")
    print(f"LPIPS: {avg_metrics['lpips']:.4f}")
    print(f"\nMetrics saved to {os.path.join(output_dir, 'metrics.json')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_ground_truth.py <lora_checkpoint_path> [test_metadata_path] [output_dir] [num_samples]")
        sys.exit(1)
    
    lora_checkpoint = sys.argv[1]
    test_metadata = sys.argv[2] if len(sys.argv) > 2 else "data/coco_video_dataset/metadata_test.csv"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "validation_outputs"
    num_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    
    validate(lora_checkpoint, test_metadata, output_dir, num_samples)
