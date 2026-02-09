import torch, os, sys, json
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import lpips

# DiffSynth 관련 import
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from examples.wanvideo.config.ground_truth_prompts import SYSTEM_PROMPT

# Pose Estimation을 위한 라이브러리 (YOLOv8-Pose 권장)
try:
    from ultralytics import YOLO
    POSE_AVAILABLE = True
except ImportError:
    POSE_AVAILABLE = False
    print("Warning: 'ultralytics' not installed. Pose metrics will be skipped.")

# ==========================================
# 1. Metric Calculation Functions
# ==========================================

def calculate_basic_metrics(generated_frame, ground_truth_frame, lpips_model):
    """기본 화질 평가: MSE, PSNR, SSIM, LPIPS"""
    gen_np = np.array(generated_frame).astype(np.float32) / 255.0
    gt_np = np.array(ground_truth_frame).astype(np.float32) / 255.0
    
    # MSE & PSNR
    mse = np.mean((gen_np - gt_np) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    # SSIM
    ssim_value = ssim(gt_np, gen_np, channel_axis=2, data_range=1.0)
    
    # LPIPS
    gen_tensor = torch.from_numpy(gen_np).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
    gt_tensor = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
    with torch.no_grad():
        lpips_value = lpips_model(gen_tensor, gt_tensor).item()
    
    return {'mse': mse, 'psnr': psnr, 'ssim': ssim_value, 'lpips': lpips_value}

def calculate_pose_metrics(generated_frame, ground_truth_frame, pose_model):
    """
    Pose 평가 (COCO 데이터셋용):
    - MPJPE (Mean Per Joint Position Error): 관절 좌표 간의 평균 유클리드 거리 (낮을수록 좋음)
    - PCK (Percentage of Correct Keypoints): 오차가 임계값 이내인 관절의 비율 (높을수록 좋음)
    """
    if pose_model is None:
        return {}

    # YOLO-Pose Inference (verbose=False로 로그 억제)
    # results[0].keypoints.xy shape: (N_people, 17, 2)
    gen_results = pose_model(generated_frame, verbose=False) 
    gt_results = pose_model(ground_truth_frame, verbose=False)

    # 사람이 감지되지 않은 경우 처리
    if not gt_results[0].keypoints.has_visible or not gen_results[0].keypoints.has_visible:
        return {'pose_mpjpe': None, 'pose_pck': None, 'pose_detected': False}

    # 가장 신뢰도 높은 사람 1명만 비교 (Single Person 가정)
    # (x, y) 좌표 추출
    gen_kpts = gen_results[0].keypoints.xy[0].cpu().numpy() # Shape: (17, 2)
    gt_kpts = gt_results[0].keypoints.xy[0].cpu().numpy()   # Shape: (17, 2)
    
    # GT에서 좌표가 (0,0)인 경우(가려짐/감지안됨)는 평가에서 제외하기 위한 마스크
    valid_mask = (gt_kpts[:, 0] > 0) & (gt_kpts[:, 1] > 0)
    
    if np.sum(valid_mask) == 0:
        return {'pose_mpjpe': None, 'pose_pck': None, 'pose_detected': False}

    # 1. MPJPE 계산 (유효한 관절만)
    diff = gen_kpts[valid_mask] - gt_kpts[valid_mask]
    dist = np.linalg.norm(diff, axis=1) # 각 관절별 거리
    mpjpe = np.mean(dist)

    # 2. PCK 계산 (임계값: 이미지 대각선 길이의 5% 또는 픽셀 상수값 예: 10px)
    # 여기서는 좀 더 관대한 pixel threshold 사용 (예: 20픽셀)
    threshold = 20.0 
    pck = np.mean(dist < threshold)

    return {
        'pose_mpjpe': float(mpjpe), # 낮을수록 좋음 (픽셀 에러)
        'pose_pck': float(pck),     # 높을수록 좋음 (정확도 %)
        'pose_detected': True
    }

# ==========================================
# 2. Main Validation Loop
# ==========================================

def validate(full_checkpoint_path, test_metadata_path, output_dir, num_samples=5):
    os.makedirs(output_dir, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- 경로 기반 모드 설정 ---
    # COCO가 경로에 있으면 Pose Metric 활성화
    is_coco_dataset = "coco" in full_checkpoint_path.lower() or "coco" in test_metadata_path.lower()
    # GPT-Edit가 경로에 있으면 Pose Metric 비활성화 (단순 화질만 봄)
    is_gptedit = "gptedit" in full_checkpoint_path.lower() or "gptedit" in test_metadata_path.lower()
    
    enable_pose = is_coco_dataset and not is_gptedit and POSE_AVAILABLE

    print(f"\n[Validation Config]")
    print(f"- Dataset Mode: {'COCO (Human Action)' if is_coco_dataset else 'General'}")
    print(f"- GPT-Edit Mode: {is_gptedit}")
    print(f"- Pose Evaluation: {'ENABLED' if enable_pose else 'DISABLED'}")
    print(f"- Checkpoint: {full_checkpoint_path}\n")

    # --- 모델 로드 ---
    # 1. LPIPS 모델 (항상 로드)
    lpips_model = lpips.LPIPS(net='alex').cuda()
    
    # 2. WanVideo 파이프라인 로드
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(path=full_checkpoint_path, offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
        ],
    )
    pipe.enable_vram_management()

    # 3. Pose Model 로드 (필요시)
    pose_estimator = None
    if enable_pose:
        print("Loading YOLOv8-Pose model for keypoint evaluation...")
        # 'yolov8n-pose.pt' (nano) or 'yolov8m-pose.pt' (medium) -> 자동 다운로드됨
        pose_estimator = YOLO('yolov8m-pose.pt') 

    # --- 데이터 처리 ---
    metadata = pd.read_csv(test_metadata_path)
    all_metrics = []

    for i in range(min(num_samples, len(metadata))):
        row = metadata.iloc[i]
        ref_img_path = row['reference_image']
        gt_img_path = row['ground_truth_image']
        prompt = row['prompt']
        
        # Prompt Cleaning
        if isinstance(prompt, str) and prompt.endswith(", "):
            prompt = prompt[:-2]
        full_prompt = SYSTEM_PROMPT + str(prompt)

        # 이미지 로드 및 리사이즈
        try:
            input_image = Image.open(ref_img_path).convert("RGB")
            # GT 이미지를 832x480으로 리사이즈 (모델 출력과 맞춤)
            ground_truth_image = Image.open(gt_img_path).convert("RGB").resize((832, 480))
        except Exception as e:
            print(f"Error loading images for sample {i}: {e}")
            continue

        print(f"Generating Sample {i+1}/{num_samples}...")
        
        # 비디오 생성
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

        # 저장
        output_path = os.path.join(output_dir, f"sample_{run_tag}_{i}.mp4")
        save_video(video, output_path, fps=15, quality=5)
        
        # 마지막 프레임 추출 및 저장
        last_frame = video[-1]
        last_frame.save(os.path.join(output_dir, f"sample_{run_tag}_{i}_last_frame.png"))
        ground_truth_image.save(os.path.join(output_dir, f"sample_{run_tag}_{i}_ground_truth.png"))

        # --- Metrics 계산 ---
        # 1. 기본 Metrics (MSE, PSNR, SSIM, LPIPS)
        metrics = calculate_basic_metrics(last_frame, ground_truth_image, lpips_model)
        
        # 2. Pose Metrics (조건부 실행)
        if enable_pose:
            pose_res = calculate_pose_metrics(last_frame, ground_truth_image, pose_estimator)
            metrics.update(pose_res)
        
        metrics['sample_id'] = i
        all_metrics.append(metrics)

        # 로그 출력 formatting
        log_msg = (f"Sample {i+1} | MSE: {metrics['mse']:.5f} | PSNR: {metrics['psnr']:.2f} | "
                   f"SSIM: {metrics['ssim']:.3f} | LPIPS: {metrics['lpips']:.3f}")
        
        if enable_pose and metrics.get('pose_detected'):
            log_msg += f" | MPJPE: {metrics['pose_mpjpe']:.2f} px | PCK: {metrics['pose_pck']:.2f}"
        
        print(log_msg)

    # --- 최종 결과 집계 및 저장 ---
    if not all_metrics:
        print("No metrics calculated.")
        return

    # 평균 계산 (None 값 제외)
    avg_metrics = {}
    keys = all_metrics[0].keys()
    
    for k in keys:
        if k == 'sample_id': continue
        values = [m[k] for m in all_metrics if m.get(k) is not None]
        if values:
            avg_metrics[k] = np.mean(values)
        else:
            avg_metrics[k] = None

    # JSON 저장
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump({'per_sample': all_metrics, 'average': avg_metrics}, f, indent=2)

    print(f"\n=== Final Average Metrics ({'COCO Pose' if enable_pose else 'Standard'}) ===")
    print(f"MSE  : {avg_metrics.get('mse', 0):.6f}")
    print(f"PSNR : {avg_metrics.get('psnr', 0):.2f} dB")
    print(f"SSIM : {avg_metrics.get('ssim', 0):.4f}")
    print(f"LPIPS: {avg_metrics.get('lpips', 0):.4f}")
    
    if enable_pose and avg_metrics.get('pose_mpjpe') is not None:
        print(f"MPJPE: {avg_metrics['pose_mpjpe']:.2f} (Lower is better)")
        print(f"PCK  : {avg_metrics['pose_pck']:.4f} (Higher is better)")
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_ground_truth_full.py <full_checkpoint_path> [test_metadata_path] [output_dir] [num_samples]")
        sys.exit(1)
    
    full_checkpoint = sys.argv[1]
    test_metadata = sys.argv[2] if len(sys.argv) > 2 else "data/coco_video_dataset/metadata_test.csv"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "validation_outputs_full"
    num_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    
    validate(full_checkpoint, test_metadata, output_dir, num_samples)