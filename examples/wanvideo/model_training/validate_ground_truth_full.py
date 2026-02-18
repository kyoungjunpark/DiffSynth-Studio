import torch, os, sys, json, re, math
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import lpips
import cv2
import torchvision
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    if not path_str:
        return None
    base = os.path.basename(path_str)
    stem = os.path.splitext(base)[0]
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else None


def coco_ann_to_kpts(ann):
    if ann is None or "keypoints" not in ann:
        return None
    kpts = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
    return kpts


def _xywh2cs(x, y, w, h, aspect_ratio, pixel_std=200):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    scale = scale * 1.25
    return center, scale


def _box2cs(box, aspect_ratio, pixel_std=200):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, aspect_ratio, pixel_std)


def get_affine_transform(center, scale, rot, output_size,
                         shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def transform_coco_kpts_to_frame(ann, ref_size, target_size):
    if ann is None or "keypoints" not in ann:
        return None
    ref_w, ref_h = ref_size
    tgt_w, tgt_h = target_size
    aspect_ratio = ref_w * 1.0 / ref_h

    center, scale = _box2cs(ann["bbox"], aspect_ratio)
    trans = get_affine_transform(center, scale, 0, [ref_w, ref_h])

    kpts = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
    for i in range(kpts.shape[0]):
        if kpts[i, 2] > 0:
            kpts[i, 0:2] = affine_transform(kpts[i, 0:2], trans)
            # scale to target size
            kpts[i, 0] = kpts[i, 0] * (tgt_w / ref_w)
            kpts[i, 1] = kpts[i, 1] * (tgt_h / ref_h)
            kpts[i, 2] = 2
        else:
            kpts[i, 2] = 0

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

def get_pose_kpts(image, model, backend="yolo"):
    img_np = np.array(image)
    if img_np.dtype == np.float32 or img_np.max() <= 1.0:
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    elif img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)

    if backend == "torchvision":
        with torch.no_grad():
            img_t = torchvision.transforms.functional.to_tensor(image).cuda()
            outputs = model([img_t])[0]
        if outputs is None or len(outputs.get("scores", [])) == 0:
            return None
        best_idx = int(outputs["scores"].argmax().item())
        kpts = outputs["keypoints"][best_idx].detach().cpu().numpy()
        return kpts

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
# 3-1. Marker OKS-based AP (color circles in GT/Gen images)
# ------------------------------------------------------------------------------
COLORS_RGB = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "gray": (128, 128, 128),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}


@dataclass
class MarkerDet:
    image_id: str
    color_idx: int
    x: float
    y: float
    score: float


@dataclass
class MarkerGT:
    image_id: str
    color_idx: int
    x: float
    y: float


def extract_markers_hsv_rgb(
    image: Image.Image,
    min_sat: float = 0.55,
    min_val: float = 0.55,
    min_area: int = 30,
    max_area: int = 2000,
    min_fill_ratio: float = 0.4,
    palette_tolerance: float = 90.0,
) -> Dict[int, List[Tuple[float, float, float]]]:
    img = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    mask = (s >= min_sat) & (v >= min_val)

    results: Dict[int, List[Tuple[float, float, float]]] = {
        i: [] for i in range(len(COLORS_RGB))
    }
    if not mask.any():
        return results

    mask_u8 = (mask.astype(np.uint8)) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    palette = np.array(list(COLORS_RGB.values()), dtype=np.float32)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        bbox_area = max(1, w * h)
        fill_ratio = area / bbox_area
        if fill_ratio < min_fill_ratio:
            continue

        component_mask = labels == label
        mean_rgb = img[component_mask].mean(axis=0)
        mean_sat = s[component_mask].mean()

        diffs = mean_rgb[None, :] - palette
        dists = np.sqrt((diffs * diffs).sum(axis=1))
        color_idx = int(dists.argmin())
        if float(dists[color_idx]) > palette_tolerance:
            continue

        cx, cy = centroids[label]
        score = float(min(1.0, mean_sat))
        results[color_idx].append((float(cx), float(cy), score))

    return results


def compute_ap(tp: List[int], fp: List[int], n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if not tp:
        return 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(1, n_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx])
    return float(ap)


def match_detections_oks(
    preds: List[MarkerDet],
    gts: List[MarkerGT],
    image_shape_by_id: Dict[str, Tuple[int, int]],
    oks_sigma: float,
    scale_mode: str,
    oks_thresh: float,
) -> Tuple[List[int], List[int]]:
    gt_by_img: Dict[str, List[MarkerGT]] = {}
    for gt in gts:
        gt_by_img.setdefault(gt.image_id, []).append(gt)

    matched = {img: np.zeros(len(pts), dtype=bool) for img, pts in gt_by_img.items()}
    preds_sorted = sorted(preds, key=lambda d: d.score, reverse=True)
    tp = []
    fp = []

    for det in preds_sorted:
        gt_list = gt_by_img.get(det.image_id, [])
        if not gt_list:
            tp.append(0)
            fp.append(1)
            continue

        h, w = image_shape_by_id.get(det.image_id, (1, 1))
        if scale_mode == "gt_bbox":
            xs = [g.x for g in gt_list]
            ys = [g.y for g in gt_list]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            bbox_area = max(1.0, (max_x - min_x + 1.0) * (max_y - min_y + 1.0))
            scale = math.sqrt(bbox_area)
        else:
            scale = float(max(h, w))

        denom = 2.0 * (max(1e-6, scale * oks_sigma) ** 2)

        best_i = -1
        best_oks = -1.0
        for i, gt in enumerate(gt_list):
            if matched[det.image_id][i]:
                continue
            d = math.hypot(det.x - gt.x, det.y - gt.y)
            oks = math.exp(-(d * d) / denom)
            if oks > best_oks:
                best_oks = oks
                best_i = i

        if best_i >= 0 and best_oks >= oks_thresh:
            matched[det.image_id][best_i] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    return tp, fp

# ------------------------------------------------------------------------------
# 4. 메인 실행
# ------------------------------------------------------------------------------
def validate(full_checkpoint_path, test_metadata_path, output_dir, num_samples=5, coco_ann_path=None):
    os.makedirs(output_dir, exist_ok=True)
    
    device = os.getenv("DS_DEVICE", "cuda")
    if device == "cuda":
        device = "cuda:0"
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
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
        pose_backend = os.getenv("POSE_BACKEND", "yolo")
        if pose_backend == "torchvision":
            print("Loading torchvision keypoint R-CNN...")
            pose_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights="DEFAULT")
            pose_model.eval().cuda()
        else:
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
    marker_preds: Dict[int, List[MarkerDet]] = {i: [] for i in range(len(COLORS_RGB))}
    marker_gts: Dict[int, List[MarkerGT]] = {i: [] for i in range(len(COLORS_RGB))}
    marker_image_shape: Dict[str, Tuple[int, int]] = {}

    print(f"\n🚀 Start Evaluation")

    for i in range(min(num_samples, len(metadata))):
        row = metadata.iloc[i]
        prompt_text = str(row['prompt'])
        
        try:
            ref_img = Image.open(row['reference_image']).convert("RGB").resize((832, 480))
            gt_img = Image.open(row['ground_truth_image']).convert("RGB").resize((832, 480))
            ref_img_pose = ref_img
        except Exception as e:
            print(f"Sample {i} Error: {e}"); continue

        # 저장 1, 2
        with open(os.path.join(output_dir, f"sample_{i}_prompt.txt"), "w") as f: f.write(prompt_text)
        ref_img.save(os.path.join(output_dir, f"sample_{i}_input_ref.png"))
        gt_img.save(os.path.join(output_dir, f"sample_{i}_target_gt.png")) # GT 저장

        use_gt_pred = os.getenv("USE_GT_PRED", "0") == "1"
        if use_gt_pred:
            last_frame = gt_img
        else:
            ref_video_frames = [ref_img] * 49
            # 비디오 생성
            video = pipe(prompt=SYSTEM_PROMPT + prompt_text, input_image=ref_img,
                         input_video=ref_video_frames,
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

            # Marker OKS-based AP (from color circles)
            image_id = f"sample_{i}"
            gt_markers = extract_markers_hsv_rgb(gt_img)
            pred_markers = extract_markers_hsv_rgb(last_frame)
            for color_idx, pts in pred_markers.items():
                for x, y, score in pts:
                    marker_preds[color_idx].append(MarkerDet(image_id, color_idx, x, y, score))
            for color_idx, pts in gt_markers.items():
                for x, y, _score in pts:
                    marker_gts[color_idx].append(MarkerGT(image_id, color_idx, x, y))
            marker_image_shape[image_id] = (gt_img.size[1], gt_img.size[0])
            
            # Pose OKS
            gen_kpts = get_pose_kpts(last_frame, pose_model, backend=pose_backend)
            ref_kpts = None

            if coco_gt is not None:
                coco_img_id = None
                if 'coco_image_id' in row and not pd.isna(row['coco_image_id']):
                    coco_img_id = int(row['coco_image_id'])
                elif 'coco_image_path' in row:
                    coco_img_id = extract_coco_image_id(row['coco_image_path'])
                if coco_img_id is None:
                    coco_img_id = extract_coco_image_id(row['reference_image']) or extract_coco_image_id(row['ground_truth_image'])
                if coco_img_id is not None and coco_img_id in coco_gt.imgs:
                    ann_ids = coco_gt.getAnnIds(imgIds=[coco_img_id], catIds=[1], iscrowd=False)
                    anns = coco_gt.loadAnns(ann_ids)
                    ann = max(anns, key=lambda a: a.get("area", 0)) if anns else None
                    # Transform COCO GT keypoints to cropped reference frame then to generated frame
                    ref_w, ref_h = Image.open(row['reference_image']).size
                    gen_w, gen_h = last_frame.size
                    ref_kpts = transform_coco_kpts_to_frame(ann, (ref_w, ref_h), (gen_w, gen_h))

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
                    ref_kpts = get_pose_kpts(ref_img_pose, pose_model, backend=pose_backend)
            else:
                ref_kpts = get_pose_kpts(ref_img_pose, pose_model, backend=pose_backend)
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

        # Marker OKS-based AP
        if any(len(v) > 0 for v in marker_gts.values()):
            thresholds_str = os.getenv("MARKER_OKS_THRESHOLDS", "0.5:0.05:0.95")
            start_s, step_s, end_s = thresholds_str.split(":")
            start = float(start_s); step = float(step_s); end = float(end_s)
            thresholds = []
            t = start
            while t <= end + 1e-9:
                thresholds.append(round(t, 6))
                t += step

            oks_sigma = float(os.getenv("MARKER_OKS_SIGMA", "0.1"))
            oks_scale = os.getenv("MARKER_OKS_SCALE", "image")

            ap_oks_by_color = {}
            color_names = list(COLORS_RGB.keys())
            for color_idx, name in enumerate(color_names):
                preds = marker_preds[color_idx]
                gts = marker_gts[color_idx]
                ap_list = []
                for thr in thresholds:
                    tp, fp = match_detections_oks(
                        preds,
                        gts,
                        image_shape_by_id=marker_image_shape,
                        oks_sigma=oks_sigma,
                        scale_mode=oks_scale,
                        oks_thresh=thr,
                    )
                    ap_list.append(compute_ap(tp, fp, n_gt=len(gts)))
                valid = [v for v in ap_list if not math.isnan(v)]
                ap_oks_by_color[name] = float(np.mean(valid)) if valid else float("nan")

            valid_oks = [v for v in ap_oks_by_color.values() if not math.isnan(v)]
            avg_metrics["marker_mAP_oks"] = float(np.mean(valid_oks)) if valid_oks else float("nan")
            avg_metrics["marker_ap_oks_by_color"] = ap_oks_by_color

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