"""Command-line entry point for the brain MRI tumour classification project.

Execution flow:
1. Resolve dataset and output paths.
2. Extract the dataset archive if the raw folder is not present.
3. Load images and infer tumour / no-tumour labels from folder names.
4. Preprocess each image, run basic segmentation, and extract HOG features.
5. Split the dataset with stratification, scale features, and train an SVM.
6. Evaluate the model and save visual outputs under outputs/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ProjectConfig
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brain MRI tumour classification using classical computer vision."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional path to the extracted dataset root.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Optional path to the dataset zip archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional path for generated figures and reports.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=4,
        help="Number of test images to visualise.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig.from_cli(
        data_root=args.data_root,
        zip_path=args.zip_path,
        output_dir=args.output_dir,
    )
    metrics = run_pipeline(config=config, sample_limit=args.sample_limit)

    print("\nFinal evaluation")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
