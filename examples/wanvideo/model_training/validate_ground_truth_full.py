import torch, os, sys, json, re
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import lpips
import cv2

# DiffSynth 관련 import
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from examples.wanvideo.config.ground_truth_prompts import SYSTEM_PROMPT

# ------------------------------------------------------------------------------
# 1. 라이브러리 체크
# ------------------------------------------------------------------------------
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: 'ultralytics' not found. Run 'pip install ultralytics'")
    sys.exit(1)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    print("❌ Error: 'pycocotools' not found. Run 'pip install pycocotools'")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 2. 평가 클래스 (Pose Preservation - JSON Safe 버전)
# ------------------------------------------------------------------------------
class PosePreservationEvaluator:
    def __init__(self):
        self.gt_data = {
            "images": [], "annotations": [],
            "categories": [{"id": 1, "name": "person", "supercategory": "person"}]
        }
        self.dt_data = []
        self.img_id = 0
        self.ann_id = 0

    def add_sample(self, ref_kpts, gen_kpts, width=832, height=480):
        self.img_id += 1
        # [수정] int() 형변환 추가
        self.gt_data["images"].append({
            "id": int(self.img_id), "width": int(width), "height": int(height), 
            "file_name": f"{self.img_id}.jpg"
        })

        # GT (Reference)
        if ref_kpts is not None:
            xs, ys = ref_kpts[:, 0], ref_kpts[:, 1]
            vis = ref_kpts[:, 2] > 0.3
            if np.sum(vis) > 0:
                # [수정] float() 형변환으로 JSON 에러 방지
                x_min, y_min = float(np.min(xs[vis])), float(np.min(ys[vis]))
                x_max, y_max = float(np.max(xs[vis])), float(np.max(ys[vis]))
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                area = float(bbox[2] * bbox[3])
            else: bbox, area = [0.0, 0.0, 0.0, 0.0], 0.0

            # COCO GT format expects v in {0,1,2}. Use v=2 for visible.
            kpts_list = []
            for kp in ref_kpts:
                v = 2 if kp[2] > 0.3 else 0
                kpts_list.extend([float(kp[0]), float(kp[1]), float(v)])
            
            self.gt_data["annotations"].append({
                "id": int(self.ann_id), "image_id": int(self.img_id), "category_id": 1,
                "keypoints": kpts_list, 
                "num_keypoints": int(np.sum(vis)),
                "area": area, "bbox": bbox, "iscrowd": 0
            })
            self.ann_id += 1

        # Pred (Generated)
        if gen_kpts is not None:
            score = np.mean(gen_kpts[:, 2])
            kpts_list = []
            for kp in gen_kpts:
                # COCO DT format expects per-keypoint scores in place of v
                kpts_list.extend([float(kp[0]), float(kp[1]), float(kp[2])])
            
            self.dt_data.append({
                "image_id": int(self.img_id), "category_id": 1,
                "keypoints": kpts_list, "score": float(score)
            })

    def evaluate(self):
        if not self.dt_data: return {}
        import tempfile
        # JSON 저장 시 에러 방지용 (default=str 추가 가능하지만 위에서 해결함)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
            json.dump(self.gt_data, tf); gt_path = tf.name
        
        cocoGt = COCO(gt_path)
        cocoDt = cocoGt.loadRes(self.dt_data)
        cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
        cocoEval.evaluate(); cocoEval.accumulate()
        print("\n" + "="*50 + "\n 🧘 Pose Preservation Results (mAP) \n" + "="*50)
        cocoEval.summarize()
        os.remove(gt_path)
        return {"mAP": float(cocoEval.stats[0]), "AP50": float(cocoEval.stats[1])}


def extract_coco_image_id(path_str: str):
    """Extract numeric COCO image id from a path like .../sample_123456.png"""
    base = os.path.basename(path_str)
    stem = os.path.splitext(base)[0]
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else None


def coco_ann_to_kpts(ann):
    if ann is None or "keypoints" not in ann:
        return None
    kpts = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
    return kpts

# ------------------------------------------------------------------------------
# 3. 도우미 함수들
# ------------------------------------------------------------------------------
def calculate_basic_metrics(gen_frame, gt_frame, lpips_model):
    gen_np = np.array(gen_frame).astype(np.float32) / 255.0
    gt_np = np.array(gt_frame).astype(np.float32) / 255.0
    
    mse = np.mean((gen_np - gt_np) ** 2)
    psnr = 100 if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))
    ssim_val = ssim(gt_np, gen_np, channel_axis=2, data_range=1.0)
    
    gen_t = torch.from_numpy(gen_np).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
    gt_t = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
    with torch.no_grad(): 
        lpips_val = lpips_model(gen_t, gt_t).item()
    
    # [수정] float() 형변환으로 JSON 저장 준비
    return {
        'mse': float(mse), 
        'psnr': float(psnr), 
        'ssim': float(ssim_val), 
        'lpips': float(lpips_val)
    }

def extract_marker_info(image, ref_image):
    """노이즈에 강건한 마커 추출"""
    img_np = np.array(image)
    ref_np = np.array(ref_image.resize(image.size))
    
    if np.mean((img_np.astype(float) - ref_np.astype(float)) ** 2) < 1.0: return None, None

    diff = cv2.absdiff(img_np, ref_np)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None
    
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10: return None, None 

    M = cv2.moments(largest)
    if M["m00"] == 0: return None, None
    
    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    
    y1, y2 = max(0, cY-2), min(img_np.shape[0], cY+3)
    x1, x2 = max(0, cX-2), min(img_np.shape[1], cX+3)
    color = np.mean(img_np[y1:y2, x1:x2], axis=(0,1))
    
    return (cX, cY), color

def calculate_marker_accuracy(gen_frame, gt_frame, ref_image):
    gt_xy, gt_rgb = extract_marker_info(gt_frame, ref_image)
    pred_xy, pred_rgb = extract_marker_info(gen_frame, ref_image)
    
    if gt_xy is None: return {'marker_detected': False, 'note': 'GT has no marker'}
    if pred_xy is None: return {'marker_detected': False, 'marker_dist_err': 999.0, 'marker_color_err': 999.0}
    
    dist_err = np.sqrt((gt_xy[0]-pred_xy[0])**2 + (gt_xy[1]-pred_xy[1])**2)
    color_err = np.sqrt(np.sum((gt_rgb - pred_rgb)**2))
    
    return {'marker_detected': True, 'marker_dist_err': float(dist_err), 'marker_color_err': float(color_err)}

def get_pose_kpts(image, model):
    img_np = np.array(image)
    if img_np.dtype == np.float32 or img_np.max() <= 1.0:
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    elif img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)

    results = model(img_np, verbose=False)
    
    # 안전장치 강화
    if not hasattr(results[0], 'keypoints') or results[0].keypoints is None: return None
    if results[0].keypoints.data.shape[0] == 0: return None
        
    return results[0].keypoints.data[0].cpu().numpy()

def calculate_instant_oks(gt_kpts, pred_kpts):
    if gt_kpts is None or pred_kpts is None: return 0.0
    sigmas = np.array([.026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089]) / 10.0
    vars = (sigmas * 2) ** 2
    
    vis_mask = gt_kpts[:, 2] > 0.3
    if np.sum(vis_mask) == 0: return 0.0
    
    xs, ys = gt_kpts[vis_mask, 0], gt_kpts[vis_mask, 1]
    area = (np.max(xs) - np.min(xs)) * (np.max(ys) - np.min(ys))
    area = max(area, 1.0)
    
    dx = gt_kpts[:, 0] - pred_kpts[:, 0]
    dy = gt_kpts[:, 1] - pred_kpts[:, 1]
    d_sq = dx**2 + dy**2
    e = d_sq / (2 * area * vars)
    return float(np.mean(np.exp(-e)[vis_mask]))

# ------------------------------------------------------------------------------
# 4. 메인 실행
# ------------------------------------------------------------------------------
def validate(full_checkpoint_path, test_metadata_path, output_dir, num_samples=5, coco_ann_path=None):
    os.makedirs(output_dir, exist_ok=True)
    
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(path=full_checkpoint_path, offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
        ]
    )
    pipe.enable_vram_management()
    
    lpips_model = lpips.LPIPS(net='alex').cuda()
    
    # 데이터셋 타입 확인 ("coco"가 포함되면 Full Evaluation)
    is_coco_dataset = "coco" in test_metadata_path.lower()
    
    pose_model = None
    pose_evaluator = None
    coco_gt = None
    dt_data_coco = []
    coco_img_ids = set()
    
    if is_coco_dataset:
        print("💡 COCO dataset detected: Running FULL Evaluation (Marker + Pose + Image Quality)")
        print("Loading YOLOv8-Pose...")
        pose_model = YOLO('yolov8m-pose.pt') 
        if coco_ann_path and os.path.exists(coco_ann_path):
            print(f"Using COCO GT annotations: {coco_ann_path}")
            coco_gt = COCO(coco_ann_path)
        else:
            pose_evaluator = PosePreservationEvaluator()
    else:
        print("💡 General dataset detected: Running Image Quality Evaluation Only (MSE, PSNR, LPIPS)")

    metadata = pd.read_csv(test_metadata_path)
    all_metrics = []

    print(f"\n🚀 Start Evaluation")

    for i in range(min(num_samples, len(metadata))):
        row = metadata.iloc[i]
        prompt_text = str(row['prompt'])
        
        try:
            ref_img = Image.open(row['reference_image']).convert("RGB")
            gt_img = Image.open(row['ground_truth_image']).convert("RGB").resize((832, 480))
            ref_img_pose = ref_img.resize((832, 480))
        except Exception as e:
            print(f"Sample {i} Error: {e}"); continue

        # 저장 1, 2
        with open(os.path.join(output_dir, f"sample_{i}_prompt.txt"), "w") as f: f.write(prompt_text)
        ref_img.save(os.path.join(output_dir, f"sample_{i}_input_ref.png"))
        gt_img.save(os.path.join(output_dir, f"sample_{i}_target_gt.png")) # GT 저장

        # 비디오 생성
        video = pipe(prompt=SYSTEM_PROMPT + prompt_text, input_image=ref_img,
                     num_frames=49, height=480, width=832, seed=0, tiled=True)
        last_frame = video[-1]
        
        # 저장 3, 4
        save_video(video, os.path.join(output_dir, f"sample_{i}.mp4"), fps=15)
        last_frame.save(os.path.join(output_dir, f"sample_{i}_output_gen.png"))

        # --- [공통] 화질 평가 (LPIPS 포함) ---
        metrics = calculate_basic_metrics(last_frame, gt_img, lpips_model)
        
        # --- [COCO 전용] Marker & Pose 평가 ---
        if is_coco_dataset:
            # Marker
            marker_res = calculate_marker_accuracy(last_frame, gt_img, ref_img)
            metrics.update(marker_res)
            
            # Pose OKS
            gen_kpts = get_pose_kpts(last_frame, pose_model)
            ref_kpts = None

            if coco_gt is not None:
                coco_img_id = extract_coco_image_id(row['reference_image']) or extract_coco_image_id(row['ground_truth_image'])
                if coco_img_id is not None and coco_img_id in coco_gt.imgs:
                    ann_ids = coco_gt.getAnnIds(imgIds=[coco_img_id], catIds=[1], iscrowd=False)
                    anns = coco_gt.loadAnns(ann_ids)
                    ann = max(anns, key=lambda a: a.get("area", 0)) if anns else None
                    ref_kpts = coco_ann_to_kpts(ann)

                    if gen_kpts is not None:
                        kpts_list = []
                        for kp in gen_kpts:
                            kpts_list.extend([float(kp[0]), float(kp[1]), float(kp[2])])
                        dt_data_coco.append({
                            "image_id": int(coco_img_id),
                            "category_id": 1,
                            "keypoints": kpts_list,
                            "score": float(np.mean(gen_kpts[:, 2]))
                        })
                        coco_img_ids.add(int(coco_img_id))
                else:
                    ref_kpts = get_pose_kpts(ref_img_pose, pose_model)
            else:
                ref_kpts = get_pose_kpts(ref_img_pose, pose_model)
            instant_oks = calculate_instant_oks(ref_kpts, gen_kpts)
            metrics['pose_oks'] = instant_oks
            
            # AP 등록 (YOLO-based GT fallback)
            if pose_evaluator is not None:
                pose_evaluator.add_sample(ref_kpts, gen_kpts, width=832, height=480)
        
        metrics['sample_id'] = i
        metrics['prompt'] = prompt_text
        metrics['reference_path'] = row['reference_image']
        
        all_metrics.append(metrics)
        
        # 로그 출력
        log = (f"Sample {i+1} | "
               f"MSE: {metrics['mse']:.5f} | "
               f"PSNR: {metrics['psnr']:.2f} | "
               f"LPIPS: {metrics['lpips']:.4f}") 
               
        if is_coco_dataset:
            log += f" | OKS: {metrics.get('pose_oks', 0.0):.3f}"
            if metrics.get('marker_detected'):
                log += (f" | Marker Dist: {metrics['marker_dist_err']:.1f}px")
            
            if metrics.get('pose_oks', 0.0) == 0.0:
                if ref_kpts is None: log += " [⚠️Ref Pose Miss]"
                if gen_kpts is None: log += " [⚠️Gen Pose Miss]"
            
        print(log)

    # 최종 결과 처리
    avg_metrics = {}
    
    # Pose AP 계산 (COCO일 때만)
    if is_coco_dataset:
        print("\nCalculating Final mAP...")
        if coco_gt is not None and dt_data_coco:
            cocoDt = coco_gt.loadRes(dt_data_coco)
            cocoEval = COCOeval(coco_gt, cocoDt, 'keypoints')
            if coco_img_ids:
                cocoEval.params.imgIds = list(coco_img_ids)
            cocoEval.evaluate(); cocoEval.accumulate(); cocoEval.summarize()
            avg_metrics.update({"mAP": float(cocoEval.stats[0]), "AP50": float(cocoEval.stats[1])})
        elif pose_evaluator is not None:
            pose_metrics = pose_evaluator.evaluate()
            avg_metrics.update(pose_metrics)

    # 나머지 평균 계산
    for k in all_metrics[0].keys():
        if isinstance(all_metrics[0][k], (int, float)) and k != 'sample_id':
            vals = [m[k] for m in all_metrics if m.get(k) is not None]
            avg_metrics[k] = float(np.mean(vals)) if vals else 0.0 # float() 변환
            
    with open(os.path.join(output_dir, 'final_metrics.json'), 'w') as f:
        json.dump({'per_sample': all_metrics, 'average': avg_metrics}, f, indent=2)

    print("\n✅ Done! Check output folder for images, prompts, and videos.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate.py <checkpoint> <metadata> <output> <samples> [coco_annotations]")
        sys.exit(1)
    validate(sys.argv[1], 
             sys.argv[2] if len(sys.argv)>2 else "data/test.csv", 
             sys.argv[3] if len(sys.argv)>3 else "output", 
             int(sys.argv[4]) if len(sys.argv)>4 else 5,
             sys.argv[5] if len(sys.argv)>5 else None)