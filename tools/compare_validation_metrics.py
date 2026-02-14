#!/usr/bin/env python3
"""Compare validation final_metrics.json across multiple folders and save plots.

Outputs:
- comparison_metrics.png: grouped bar charts for each metric
- comparison_metrics.csv: flat table of averages per run
- missing_runs.txt: list of runs missing final_metrics.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import math

BASE_DIR = Path("/home/aiscuser/DiffSynth-Studio/validation_0214")

RUN_PREFIX = "validation_outputs_"

METRICS = [
    "mAP",
    "AP50",
    "marker_mAP_oks",
    "pose_oks",
    "mse",
    "psnr",
    "ssim",
    "lpips",
    "marker_detected",
    "marker_dist_err",
    "marker_color_err",
    "marker_ap_oks_by_color.red",
    "marker_ap_oks_by_color.green",
    "marker_ap_oks_by_color.blue",
    "marker_ap_oks_by_color.yellow",
    "marker_ap_oks_by_color.cyan",
    "marker_ap_oks_by_color.magenta",
    "marker_ap_oks_by_color.gray",
    "marker_ap_oks_by_color.white",
    "marker_ap_oks_by_color.black",
]


def discover_run_folders(base_dir: Path) -> List[str]:
    return sorted(
        [
            p.name
            for p in base_dir.iterdir()
            if p.is_dir() and p.name.startswith(RUN_PREFIX) and p.name.endswith("steps")
        ]
    )


def load_averages(base_dir: Path, folders: List[str]) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    for folder in folders:
        path = base_dir / folder / "final_metrics.json"
        if not path.exists():
            continue
        with path.open() as f:
            data = json.load(f)
        avg = data.get("average", {})
        row: Dict[str, float] = {}
        for key in METRICS:
            if "." in key:
                root, leaf = key.split(".", 1)
                value = avg.get(root, {}) if isinstance(avg.get(root, {}), dict) else {}
                row[key] = value.get(leaf)
            else:
                row[key] = avg.get(key)
        rows[folder] = row
    return rows


def save_csv(rows: Dict[str, Dict[str, float]], out_path: Path) -> None:
    headers = ["name"] + METRICS
    lines = [",".join(headers)]
    for name, vals in rows.items():
        line = [name] + ["" if vals.get(k) is None else str(vals.get(k)) for k in METRICS]
        lines.append(",".join(line))
    out_path.write_text("\n".join(lines))


def plot_metrics(rows: Dict[str, Dict[str, float]], out_path: Path) -> None:
    names = list(rows.keys())
    if not names:
        raise RuntimeError("No runs with final_metrics.json found.")

    # Group by methodology (folder name without trailing _<steps>steps)
    # Example: validation_outputs_..._10000steps -> method name
    grouped: Dict[str, Dict[int, Dict[str, float]]] = {}
    for name, metrics in rows.items():
        step = None
        if name.endswith("steps"):
            try:
                step = int(name.split("_")[-1].replace("steps", ""))
            except ValueError:
                step = None
        method = name
        if step is not None:
            method = name[: -(len(str(step)) + len("steps") + 1)]

        grouped.setdefault(method, {})[step if step is not None else -1] = metrics

    # Sort methods for consistency
    methods = sorted(grouped.keys())

    n_metrics = len(METRICS)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), squeeze=False)

    for idx, metric in enumerate(METRICS):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]

        for method in methods:
            steps = sorted(grouped[method].keys())
            values = []
            for s in steps:
                v = grouped[method][s].get(metric)
                values.append(math.nan if v is None else v)
            ax.plot(steps, values, marker="o", linewidth=1.5, label=method)

        ax.set_title(metric)
        ax.set_xlabel("steps")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Hide unused axes if any
    for idx in range(n_metrics, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].axis("off")

    # Add legend outside the plot for readability
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=1, fontsize=8, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    base_dir = BASE_DIR
    out_dir = base_dir / "analysis_outputs"
    out_dir.mkdir(exist_ok=True)

    run_folders = discover_run_folders(base_dir)
    rows = load_averages(base_dir, run_folders)

    # Save missing runs list
    missing = [f for f in run_folders if not (base_dir / f / "final_metrics.json").exists()]
    (out_dir / "missing_runs.txt").write_text("\n".join(missing))

    # Save CSV and plot
    save_csv(rows, out_dir / "comparison_metrics.csv")
    plot_metrics(rows, out_dir / "comparison_metrics.png")

    coco_rows = {k: v for k, v in rows.items() if "coco" in k}
    gptedit_rows = {k: v for k, v in rows.items() if "gptedit" in k}
    if coco_rows:
        plot_metrics(coco_rows, out_dir / "comparison_metrics_coco.png")
    if gptedit_rows:
        plot_metrics(gptedit_rows, out_dir / "comparison_metrics_gptedit.png")

    print("Saved:")
    print(out_dir / "comparison_metrics.csv")
    print(out_dir / "comparison_metrics.png")
    if coco_rows:
        print(out_dir / "comparison_metrics_coco.png")
    if gptedit_rows:
        print(out_dir / "comparison_metrics_gptedit.png")
    print(out_dir / "missing_runs.txt")


if __name__ == "__main__":
    main()
