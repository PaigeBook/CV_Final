"""Visual output helpers for the academic report artifacts."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def ensure_output_structure(output_dir: Path) -> dict[str, Path]:
    figures_dir = output_dir / "figures"
    cases_dir = output_dir / "cases"
    reports_dir = output_dir / "reports"
    for directory in (output_dir, figures_dir, cases_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "figures": figures_dir,
        "cases": cases_dir,
        "reports": reports_dir,
    }


def save_confusion_matrix(confusion_matrix_values: list[list[int]], output_path: Path) -> None:
    matrix = np.asarray(confusion_matrix_values)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No tumour", "Tumour"])
    ax.set_yticklabels(["No tumour", "Tumour"])

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, int(matrix[row_index, col_index]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_case_artifacts(
    original_image: np.ndarray,
    enhanced_image: np.ndarray,
    segmentation_mask: np.ndarray,
    contour_overlay: np.ndarray,
    predicted_label: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original MRI")
    axes[0].axis("off")

    axes[1].imshow(enhanced_image, cmap="gray")
    axes[1].set_title("Enhanced image")
    axes[1].axis("off")

    axes[2].imshow(segmentation_mask, cmap="gray")
    axes[2].set_title("Segmentation mask")
    axes[2].axis("off")

    axes[3].imshow(cv2.cvtColor(contour_overlay, cv2.COLOR_BGR2RGB))
    axes[3].set_title(f"Contour overlay\nPrediction: {predicted_label}")
    axes[3].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_case_comparison_grid(
    case_items: list[tuple[np.ndarray, str, str]],
    output_path: Path,
) -> None:
    """Save a 2x2 overview of the sampled MRI cases for quick visual comparison."""

    if not case_items:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for axis, (original_image, actual_label, predicted_label) in zip(axes, case_items):
        axis.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axis.set_title(f"Actual: {actual_label}\nPredicted: {predicted_label}")
        axis.axis("off")

    for axis in axes[len(case_items):]:
        axis.axis("off")

    fig.suptitle("Random MRI Case Comparison", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
