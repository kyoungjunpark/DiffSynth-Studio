#!/usr/bin/env python3
"""Compute mAP for keypoint markers drawn as colored circles in images.

This script extracts marker centers by color from both predicted and GT images,
then computes AP per color and mAP across colors.

Example input folder layout:
- sample_0_output_gen.png
- sample_0_target_gt.png

Assumes marker colors follow the palette in tools/build_coco_metadata.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv


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
class Detection:
    image_id: str
    color_idx: int
    x: float
    y: float
    score: float


@dataclass
class GTPoint:
    image_id: str
    color_idx: int
    x: float
    y: float


@dataclass
class Point2D:
    x: float
    y: float


def list_samples(folder: Path, pred_suffix: str, gt_suffix: str) -> List[Tuple[str, Path, Path]]:
    pred_files = sorted(folder.glob(f"*{pred_suffix}"))
    samples = []
    for pred_path in pred_files:
        base = pred_path.name[: -len(pred_suffix)]
        gt_path = folder / f"{base}{gt_suffix}"
        if gt_path.exists():
            samples.append((base, pred_path, gt_path))
    return samples


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def extract_markers_palette(
    image: np.ndarray,
    palette: np.ndarray,
    tolerance: float,
    min_area: int,
    max_area: int,
    min_fill_ratio: float,
) -> Dict[int, List[Tuple[float, float, float]]]:
    """Return per-color list of (x, y, score) marker detections."""
    h, w, _ = image.shape
    flat = image.reshape(-1, 3).astype(np.int32)
    palette_i = palette.astype(np.int32)

    # Compute squared distance to palette colors
    diffs = flat[:, None, :] - palette_i[None, :, :]
    dists = (diffs * diffs).sum(axis=2)
    min_dist = dists.min(axis=1)
    min_idx = dists.argmin(axis=1)

    tol2 = tolerance * tolerance
    valid = min_dist <= tol2
    assign = min_idx.reshape(h, w)
    dist_map = np.sqrt(min_dist).reshape(h, w)
    valid_map = valid.reshape(h, w)

    results: Dict[int, List[Tuple[float, float, float]]] = {i: [] for i in range(len(palette))}

    for color_idx in range(len(palette)):
        mask = (assign == color_idx) & valid_map
        if not mask.any():
            continue

        visited = np.zeros((h, w), dtype=bool)
        ys, xs = np.where(mask)
        for y0, x0 in zip(ys, xs):
            if visited[y0, x0]:
                continue
            # flood fill
            stack = [(y0, x0)]
            visited[y0, x0] = True
            area = 0
            sum_x = 0.0
            sum_y = 0.0
            min_x = x0
            max_x = x0
            min_y = y0
            max_y = y0
            sum_dist = 0.0

            while stack:
                y, x = stack.pop()
                area += 1
                sum_x += x
                sum_y += y
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
                sum_dist += float(dist_map[y, x])

                for ny in range(y - 1, y + 2):
                    if ny < 0 or ny >= h:
                        continue
                    for nx in range(x - 1, x + 2):
                        if nx < 0 or nx >= w:
                            continue
                        if not visited[ny, nx] and mask[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            if area < min_area or area > max_area:
                continue

            bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
            fill_ratio = area / max(1, bbox_area)
            if fill_ratio < min_fill_ratio:
                continue

            cx = sum_x / area
            cy = sum_y / area
            mean_dist = sum_dist / area
            score = max(0.0, 1.0 - (mean_dist / max(1e-6, tolerance)))
            results[color_idx].append((cx, cy, score))

    return results


def extract_markers_hsv(
    image: np.ndarray,
    palette: np.ndarray,
    min_sat: float,
    min_val: float,
    min_area: int,
    max_area: int,
    min_fill_ratio: float,
    palette_tolerance: float,
) -> Dict[int, List[Tuple[float, float, float]]]:
    h, w, _ = image.shape
    img_float = image.astype(np.float32) / 255.0
    hsv = rgb_to_hsv(img_float)
    mask = (hsv[..., 1] >= min_sat) & (hsv[..., 2] >= min_val)

    results: Dict[int, List[Tuple[float, float, float]]] = {i: [] for i in range(len(palette))}
    if not mask.any():
        return results

    visited = np.zeros((h, w), dtype=bool)
    ys, xs = np.where(mask)
    palette_i = palette.astype(np.int32)

    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        area = 0
        sum_x = 0.0
        sum_y = 0.0
        min_x = x0
        max_x = x0
        min_y = y0
        max_y = y0
        sum_rgb = np.zeros(3, dtype=np.float64)
        sum_sat = 0.0

        while stack:
            y, x = stack.pop()
            area += 1
            sum_x += x
            sum_y += y
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            sum_rgb += image[y, x]
            sum_sat += float(hsv[y, x, 1])

            for ny in range(y - 1, y + 2):
                if ny < 0 or ny >= h:
                    continue
                for nx in range(x - 1, x + 2):
                    if nx < 0 or nx >= w:
                        continue
                    if not visited[ny, nx] and mask[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

        if area < min_area or area > max_area:
            continue

        bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
        fill_ratio = area / max(1, bbox_area)
        if fill_ratio < min_fill_ratio:
            continue

        mean_rgb = sum_rgb / area
        diffs = mean_rgb[None, :] - palette_i
        dists = np.sqrt((diffs * diffs).sum(axis=1))
        color_idx = int(dists.argmin())
        if float(dists[color_idx]) > palette_tolerance:
            continue

        cx = sum_x / area
        cy = sum_y / area
        score = min(1.0, sum_sat / area)
        results[color_idx].append((cx, cy, score))

    return results


def match_detections(
    preds: List[Detection],
    gts: List[GTPoint],
    match_dist: float,
) -> Tuple[List[int], List[int]]:
    # group GT by image
    gt_by_img: Dict[str, List[GTPoint]] = {}
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

        best_i = -1
        best_d = float("inf")
        for i, gt in enumerate(gt_list):
            if matched[det.image_id][i]:
                continue
            d = math.hypot(det.x - gt.x, det.y - gt.y)
            if d < best_d:
                best_d = d
                best_i = i

        if best_i >= 0 and best_d <= match_dist:
            matched[det.image_id][best_i] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    return tp, fp


def match_detections_oks(
    preds: List[Detection],
    gts: List[GTPoint],
    image_shape_by_id: Dict[str, Tuple[int, int]],
    oks_sigma: float,
    scale_mode: str,
    oks_thresh: float,
) -> Tuple[List[int], List[int]]:
    gt_by_img: Dict[str, List[GTPoint]] = {}
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


def compute_oks_for_image(
    gt_by_color: Dict[int, List[Point2D]],
    pred_by_color: Dict[int, List[Point2D]],
    image_shape: Tuple[int, int],
    oks_sigma: float,
    scale_mode: str,
) -> float:
    h, w = image_shape
    gt_points: List[Point2D] = []
    for pts in gt_by_color.values():
        gt_points.extend(pts)

    if not gt_points:
        return float("nan")

    if scale_mode == "gt_bbox":
        xs = [p.x for p in gt_points]
        ys = [p.y for p in gt_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bbox_area = max(1.0, (max_x - min_x + 1.0) * (max_y - min_y + 1.0))
        scale = math.sqrt(bbox_area)
    else:
        scale = float(max(h, w))

    denom = 2.0 * (max(1e-6, scale * oks_sigma) ** 2)
    oks_vals: List[float] = []

    for color_idx, gt_list in gt_by_color.items():
        preds = pred_by_color.get(color_idx, [])
        for gt in gt_list:
            if not preds:
                oks_vals.append(0.0)
                continue
            best_d = min(math.hypot(gt.x - pr.x, gt.y - pr.y) for pr in preds)
            oks_vals.append(math.exp(-(best_d * best_d) / denom))

    if not oks_vals:
        return float("nan")
    return float(sum(oks_vals) / len(oks_vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute keypoint mAP from marker images.")
    parser.add_argument(
        "--folder",
        type=str,
        default="/home/aiscuser/DiffSynth-Studio/vadliation_0212/validation_outputs_gptedit_Wan2.2-TI2V-5B_full_two_dataset_10000steps",
        help="Folder containing predicted/GT images.",
    )
    parser.add_argument("--pred-suffix", type=str, default="_output_gen.png")
    parser.add_argument("--gt-suffix", type=str, default="_target_gt.png")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["palette", "hsv"],
        default="hsv",
        help="Marker extraction mode.",
    )
    parser.add_argument("--tolerance", type=float, default=40.0, help="RGB distance tolerance for palette mode.")
    parser.add_argument("--min-sat", type=float, default=0.55, help="Minimum saturation for HSV mode.")
    parser.add_argument("--min-val", type=float, default=0.55, help="Minimum value for HSV mode.")
    parser.add_argument(
        "--palette-tolerance",
        type=float,
        default=90.0,
        help="Max palette RGB distance for HSV mode color assignment.",
    )
    parser.add_argument("--min-area", type=int, default=30, help="Minimum blob area in pixels.")
    parser.add_argument("--max-area", type=int, default=2000, help="Maximum blob area in pixels.")
    parser.add_argument("--min-fill-ratio", type=float, default=0.4, help="Minimum fill ratio for blob bbox.")
    parser.add_argument("--match-dist", type=float, default=12.0, help="Max distance for a correct match.")
    parser.add_argument("--compute-oks", action="store_true", help="Compute OKS in addition to mAP.")
    parser.add_argument("--oks-sigma", type=float, default=0.1, help="OKS sigma (normalized).")
    parser.add_argument(
        "--oks-scale",
        type=str,
        choices=["image", "gt_bbox"],
        default="image",
        help="OKS scale mode.",
    )
    parser.add_argument(
        "--oks-ap",
        action="store_true",
        help="Compute OKS-based AP (mean over thresholds).",
    )
    parser.add_argument(
        "--oks-thresholds",
        type=str,
        default="0.5:0.05:0.95",
        help="OKS thresholds as start:step:end (inclusive).",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of samples (0 = all).")
    parser.add_argument("--output", type=str, default="", help="Output JSON path.")

    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    samples = list_samples(folder, args.pred_suffix, args.gt_suffix)
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]

    if not samples:
        raise RuntimeError("No matched sample pairs found.")

    palette = np.array(list(COLORS_RGB.values()), dtype=np.int32)
    color_names = list(COLORS_RGB.keys())

    all_preds: Dict[int, List[Detection]] = {i: [] for i in range(len(palette))}
    all_gts: Dict[int, List[GTPoint]] = {i: [] for i in range(len(palette))}
    per_image_pred: Dict[str, Dict[int, List[Point2D]]] = {}
    per_image_gt: Dict[str, Dict[int, List[Point2D]]] = {}
    per_image_shape: Dict[str, Tuple[int, int]] = {}

    for base, pred_path, gt_path in samples:
        pred_img = load_image(pred_path)
        gt_img = load_image(gt_path)

        if args.mode == "palette":
            pred_markers = extract_markers_palette(
                pred_img,
                palette,
                tolerance=args.tolerance,
                min_area=args.min_area,
                max_area=args.max_area,
                min_fill_ratio=args.min_fill_ratio,
            )
            gt_markers = extract_markers_palette(
                gt_img,
                palette,
                tolerance=args.tolerance,
                min_area=args.min_area,
                max_area=args.max_area,
                min_fill_ratio=args.min_fill_ratio,
            )
        else:
            pred_markers = extract_markers_hsv(
                pred_img,
                palette,
                min_sat=args.min_sat,
                min_val=args.min_val,
                min_area=args.min_area,
                max_area=args.max_area,
                min_fill_ratio=args.min_fill_ratio,
                palette_tolerance=args.palette_tolerance,
            )
            gt_markers = extract_markers_hsv(
                gt_img,
                palette,
                min_sat=args.min_sat,
                min_val=args.min_val,
                min_area=args.min_area,
                max_area=args.max_area,
                min_fill_ratio=args.min_fill_ratio,
                palette_tolerance=args.palette_tolerance,
            )

        for color_idx, pts in pred_markers.items():
            for x, y, score in pts:
                all_preds[color_idx].append(Detection(base, color_idx, x, y, score))
                per_image_pred.setdefault(base, {}).setdefault(color_idx, []).append(Point2D(x, y))
        for color_idx, pts in gt_markers.items():
            for x, y, _score in pts:
                all_gts[color_idx].append(GTPoint(base, color_idx, x, y))
                per_image_gt.setdefault(base, {}).setdefault(color_idx, []).append(Point2D(x, y))

        per_image_shape[base] = (gt_img.shape[0], gt_img.shape[1])

    ap_by_color: Dict[str, float] = {}
    for color_idx, name in enumerate(color_names):
        preds = all_preds[color_idx]
        gts = all_gts[color_idx]
        tp, fp = match_detections(preds, gts, match_dist=args.match_dist)
        ap = compute_ap(tp, fp, n_gt=len(gts))
        ap_by_color[name] = ap

    valid_aps = [v for v in ap_by_color.values() if not math.isnan(v)]
    mAP = float(np.mean(valid_aps)) if valid_aps else float("nan")

    pred_counts = {color_names[i]: len(all_preds[i]) for i in range(len(color_names))}
    gt_counts = {color_names[i]: len(all_gts[i]) for i in range(len(color_names))}

    output = {
        "folder": str(folder),
        "num_samples": len(samples),
        "settings": {
            "mode": args.mode,
            "tolerance": args.tolerance,
            "min_sat": args.min_sat,
            "min_val": args.min_val,
            "palette_tolerance": args.palette_tolerance,
            "min_area": args.min_area,
            "max_area": args.max_area,
            "min_fill_ratio": args.min_fill_ratio,
            "match_dist": args.match_dist,
            "compute_oks": args.compute_oks,
            "oks_sigma": args.oks_sigma,
            "oks_scale": args.oks_scale,
            "oks_ap": args.oks_ap,
            "oks_thresholds": args.oks_thresholds,
        },
        "pred_counts": pred_counts,
        "gt_counts": gt_counts,
        "ap_by_color": ap_by_color,
        "mAP": mAP,
    }

    if args.compute_oks:
        oks_vals = []
        for image_id in per_image_shape.keys():
            gt_by_color = per_image_gt.get(image_id, {})
            pred_by_color = per_image_pred.get(image_id, {})
            oks = compute_oks_for_image(
                gt_by_color,
                pred_by_color,
                per_image_shape[image_id],
                oks_sigma=args.oks_sigma,
                scale_mode=args.oks_scale,
            )
            if not math.isnan(oks):
                oks_vals.append(oks)

        output["mean_oks"] = float(np.mean(oks_vals)) if oks_vals else float("nan")

    if args.oks_ap:
        try:
            start_s, step_s, end_s = args.oks_thresholds.split(":")
            start = float(start_s)
            step = float(step_s)
            end = float(end_s)
        except ValueError as exc:
            raise ValueError("--oks-thresholds must be start:step:end") from exc

        thresholds = []
        t = start
        while t <= end + 1e-9:
            thresholds.append(round(t, 6))
            t += step

        ap_oks_by_color: Dict[str, float] = {}
        for color_idx, name in enumerate(color_names):
            preds = all_preds[color_idx]
            gts = all_gts[color_idx]
            ap_list = []
            for thr in thresholds:
                tp, fp = match_detections_oks(
                    preds,
                    gts,
                    image_shape_by_id=per_image_shape,
                    oks_sigma=args.oks_sigma,
                    scale_mode=args.oks_scale,
                    oks_thresh=thr,
                )
                ap_list.append(compute_ap(tp, fp, n_gt=len(gts)))
            valid = [v for v in ap_list if not math.isnan(v)]
            ap_oks_by_color[name] = float(np.mean(valid)) if valid else float("nan")

        valid_oks_aps = [v for v in ap_oks_by_color.values() if not math.isnan(v)]
        output["ap_oks_by_color"] = ap_oks_by_color
        output["mAP_oks"] = float(np.mean(valid_oks_aps)) if valid_oks_aps else float("nan")

    out_path = Path(args.output) if args.output else folder / "marker_map.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved: {out_path}")
    print(f"mAP: {mAP:.4f}")
    if args.compute_oks:
        print(f"mean OKS: {output.get('mean_oks', float('nan')):.4f}")
    if args.oks_ap:
        print(f"mAP (OKS): {output.get('mAP_oks', float('nan')):.4f}")


if __name__ == "__main__":
    main()
