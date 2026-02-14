#!/usr/bin/env python3
"""Build COCO-based metadata CSVs without using ds2_metadata.json.

This script directly instantiates the InstructDiffusion COCO pose dataset to
produce prompts and ground-truth edited images, while linking reference images
to the real COCO image files.

Outputs:
- metadata_train_final.csv
- metadata_val_final.csv
- ground_truth_train2017/ (edited target images)
- ground_truth_val2017/ (edited target images)
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import cv2
from pycocotools.coco import COCO

INSTRUCT_DIFFUSION_ROOT = Path("/blob/kyoungjun/InstructDiffusion")
COCO_ROOT = INSTRUCT_DIFFUSION_ROOT / "data" / "coco"

# Ensure relative prompt paths resolve correctly inside InstructDiffusion
os.chdir(str(INSTRUCT_DIFFUSION_ROOT))
sys.path.append(str(INSTRUCT_DIFFUSION_ROOT))

from dataset.pose.pose import COCODataset  # noqa: E402

KEYPOINTS_TYPE = {
    0: "nose",
    1: "left eye",
    2: "right eye",
    3: "left ear",
    4: "right ear",
    5: "left shoulder",
    6: "right shoulder",
    7: "left elbow",
    8: "right elbow",
    9: "left wrist",
    10: "right wrist",
    11: "left hip",
    12: "right hip",
    13: "left knee",
    14: "right knee",
    15: "left ankle",
    16: "right ankle",
}

COLORS = {
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    # t: CxHxW, range [-1, 1]
    t = t.detach().cpu().float()
    t = (t + 1) * 127.5
    t = t.clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(t)


def load_prompt_templates() -> list[str]:
    prompt_path = INSTRUCT_DIFFUSION_ROOT / "dataset" / "prompt" / "prompt_pose.txt"
    with open(prompt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def draw_keypoints_on_image(img: Image.Image, kpts: np.ndarray, radius: int, prompt_list: list[str],
                            min_prompt_num: int, max_prompt_num: int) -> tuple[Image.Image, str]:
    img_np = np.array(img)
    target = img_np.copy()

    joint_num = random.randint(min_prompt_num, max_prompt_num)
    joint_ids = np.random.choice([i for i in range(len(KEYPOINTS_TYPE))], joint_num, replace=False)
    random_color_names = random.sample(list(COLORS.keys()), len(joint_ids))

    prompt = ""
    for color_idx, joint_id in enumerate(joint_ids):
        x, y, v = kpts[joint_id]
        color_name = random_color_names[color_idx]
        prompt += random.choice(prompt_list).format(color=color_name, joint=KEYPOINTS_TYPE[joint_id])

        if v > 0.5:
            cv2.circle(target, (int(x), int(y)), radius, COLORS[color_name], thickness=-1)

    return Image.fromarray(target), prompt


def build_split(split: str, output_dir: Path, max_samples: int | None, size: int, use_original: bool) -> Path:
    if use_original:
        coco_ann = COCO_ROOT / "annotations" / f"person_keypoints_{split}.json"
        coco = COCO(str(coco_ann))
        prompt_list = load_prompt_templates()

        gt_dir = output_dir / f"ground_truth_{split}"
        ref_dir = output_dir / f"reference_{split}"
        gt_dir.mkdir(parents=True, exist_ok=True)
        ref_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / f"metadata_{'train' if 'train' in split else 'val'}_final.csv"
        rows = []

        img_ids = coco.getImgIds()
        max_count = len(img_ids) if max_samples is None else max_samples
        pbar = tqdm(total=max_count, desc=f"Building {split} (original)")

        idx = 0
        for img_id in img_ids:
            if len(rows) >= max_count:
                break
            img_info = coco.loadImgs([img_id])[0]
            file_name = img_info["file_name"]
            ref_path = (COCO_ROOT / "images" / split / file_name).resolve()
            if not ref_path.exists():
                nested = COCO_ROOT / "images" / split / split / file_name
                if nested.exists():
                    ref_path = nested.resolve()
                else:
                    continue

            ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[1], iscrowd=False)
            anns = coco.loadAnns(ann_ids)
            ann = max(anns, key=lambda a: a.get("area", 0)) if anns else None
            if ann is None or "keypoints" not in ann:
                continue

            kpts = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)

            ref_img = Image.open(ref_path).convert("RGB")
            gt_img, prompt = draw_keypoints_on_image(ref_img, kpts, radius=10, prompt_list=prompt_list,
                                                     min_prompt_num=1, max_prompt_num=5)

            ref_save_path = ref_dir / f"sample_{idx:06d}.png"
            gt_path = gt_dir / f"sample_{idx:06d}.png"
            ref_img.save(ref_save_path)
            gt_img.save(gt_path)

            rows.append({
                "prompt": prompt.strip(),
                "reference_image": str(ref_save_path.resolve()),
                "ground_truth_image": str(gt_path.resolve()),
                "coco_image_id": int(img_id),
                "coco_image_path": str(ref_path),
            })
            idx += 1
            pbar.update(1)

        pbar.close()

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "prompt",
                    "reference_image",
                    "ground_truth_image",
                    "coco_image_id",
                    "coco_image_path",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        return csv_path

    dataset = COCODataset(
        root=str(COCO_ROOT),
        image_set=split,
        is_train=False,
        max_prompt_num=5,
        min_prompt_num=1,
        radius=10,
        size=size,
        transparency=0.0,
        sample_weight=1.0,
        transform=None,
    )

    gt_dir = output_dir / f"ground_truth_{split}"
    ref_dir = output_dir / f"reference_{split}"
    gt_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"metadata_{'train' if 'train' in split else 'val'}_final.csv"

    rows = []
    max_count = len(dataset) if max_samples is None else max_samples

    # Fix nested val2017/val2017 path layout before accessing dataset[idx]
    if split == "val2017":
        nested_dir = COCO_ROOT / "images" / "val2017" / "val2017"
        if nested_dir.exists():
            for rec in dataset.db:
                img_path = rec.get("image")
                if img_path and "/images/val2017/" in img_path:
                    rec["image"] = img_path.replace("/images/val2017/", "/images/val2017/val2017/")

    pbar = tqdm(total=max_count, desc=f"Building {split}")
    idx = 0
    while idx < len(dataset) and len(rows) < max_count:
        ref_path = Path(dataset.db[idx]["image"]).resolve()
        if not ref_path.exists():
            idx += 1
            continue

        sample = dataset[idx]
        prompt = sample["edit"]["c_crossattn"].strip()

        # Save reference image after the same affine/crop pipeline as GT
        ref_img = tensor_to_pil(sample["edit"]["c_concat"])
        ref_save_path = ref_dir / f"sample_{idx:06d}.png"
        ref_img.save(ref_save_path)

        gt_img = tensor_to_pil(sample["edited"])
        gt_path = gt_dir / f"sample_{idx:06d}.png"
        gt_img.save(gt_path)

        coco_image_id = int(os.path.splitext(ref_path.name)[0]) if ref_path.name[:1].isdigit() else None

        rows.append({
            "prompt": prompt,
            "reference_image": str(ref_save_path.resolve()),
            "ground_truth_image": str(gt_path.resolve()),
            "coco_image_id": coco_image_id,
            "coco_image_path": str(ref_path),
        })
        pbar.update(1)
        idx += 1
    pbar.close()

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt",
                "reference_image",
                "ground_truth_image",
                "coco_image_id",
                "coco_image_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="/home/aiscuser/DiffSynth-Studio/data/coco_video_dataset")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--size", type=int, default=704)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="both", choices=["train2017", "val2017", "both"])
    parser.add_argument("--use-original", action="store_true", help="Use original COCO resolution without crop/resize")
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.split in ("train2017", "both"):
        train_csv = build_split("train2017", output_dir, args.max_samples, args.size, args.use_original)
        print(f"Saved: {train_csv}")

    if args.split in ("val2017", "both"):
        try:
            val_csv = build_split("val2017", output_dir, args.max_samples, args.size, args.use_original)
            print(f"Saved: {val_csv}")
        except Exception as e:
            print(f"Skipped val2017 due to error: {e}")


if __name__ == "__main__":
    main()
