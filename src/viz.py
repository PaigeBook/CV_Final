"""High-level visualisation helpers for the MRI classification pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog

from .classifier import PreprocessingSettings, infer_label_from_path, predict_image
from .preprocessing import load_original_image, preprocess_image
from .segmentation import overlay_mask, segment_suspicious_region
from .visualization import save_case_artifacts


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _build_preprocessing_settings(clahe_clip: float, clahe_grid: tuple[int, int]) -> PreprocessingSettings:
    return PreprocessingSettings(
        image_size=(256, 256),
        gaussian_kernel=(5, 5),
        clahe_clip_limit=float(clahe_clip),
        clahe_tile_grid_size=tuple(int(value) for value in clahe_grid),
    )


def overlay_label(
    image: np.ndarray,
    predicted_label: str,
    *,
    ground_truth_label: str | None = None,
    confidence: float | None = None,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Draw a confidence banner on top of an MRI image."""

    output = _ensure_bgr(image)
    banner = output.copy()
    cv2.rectangle(banner, (0, 0), (banner.shape[1], 44), (0, 0, 0), thickness=-1)
    text = f"Predicted: {predicted_label}"
    if ground_truth_label is not None:
        text += f" | Ground truth: {ground_truth_label}"
    if confidence is not None:
        text += f" | Confidence: {confidence:.3f}"
    cv2.putText(banner, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), banner)

    return banner


def save_visual_summary(
    original_image: np.ndarray,
    enhanced_image: np.ndarray,
    segmentation_mask: np.ndarray,
    contour_overlay: np.ndarray,
    predicted_label: str,
    *,
    ground_truth_label: str | None = None,
    confidence: float | None = None,
    output_path: str | Path,
) -> Path:
    """Save a compact 2x2 visual summary for report figures."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) if original_image.ndim == 3 else original_image, cmap="gray")
    axes[0].set_title("Original MRI")
    axes[0].axis("off")

    axes[1].imshow(enhanced_image, cmap="gray")
    axes[1].set_title("Enhanced image")
    axes[1].axis("off")

    axes[2].imshow(segmentation_mask, cmap="gray")
    axes[2].set_title("Segmentation mask")
    axes[2].axis("off")

    axes[3].imshow(cv2.cvtColor(contour_overlay, cv2.COLOR_BGR2RGB))
    title = f"Predicted: {predicted_label}"
    if ground_truth_label is not None:
        title += f"\nGround truth: {ground_truth_label}"
    if confidence is not None:
        title += f"\nConfidence: {confidence:.3f}"
    axes[3].set_title(title)
    axes[3].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _gallery_folder(predicted_label: str, ground_truth_label: str | None) -> str:
    if ground_truth_label is not None and predicted_label == ground_truth_label:
        return "correct"
    return "incorrect"


def save_annotated_predictions(
    model: object,
    sample_paths: list[Path],
    output_dir: str | Path,
    figures_dir: str | Path,
    *,
    hog_parameters,
    segmentation_method: str = "mri_enhanced",
    use_glcm: bool = False,
    use_orb: bool = False,
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_grid: tuple[int, int] = (8, 8),
    morph_kernel_size: int = 5,
    ground_truth_root: Path | None = None,
) -> list[Path]:
    """Save a small set of prediction composites grouped by outcome."""

    del figures_dir  # The pipeline keeps the report assets under the gallery tree.
    del segmentation_method  # The classical segmentation path is fixed in this project.
    del use_clahe  # Kept for CLI compatibility.

    output_dir = Path(output_dir)
    folders = {
        "correct": output_dir / "correct",
        "incorrect": output_dir / "incorrect",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    preprocessing_settings = _build_preprocessing_settings(clahe_clip, clahe_grid)
    saved_paths: list[Path] = []

    for index, image_path in enumerate(sample_paths):
        original_image = load_original_image(image_path)
        enhanced_image = preprocess_image(original_image, preprocessing_settings)
        segmentation_mask, contour_overlay, suspicious_mask = segment_suspicious_region(enhanced_image, morph_kernel_size)
        masked_image = enhanced_image.copy()
        masked_image[suspicious_mask == 0] = 0

        predicted_label, confidence = predict_image(
            model,
            masked_image,
            hog_parameters=hog_parameters,
            use_glcm=use_glcm,
            use_orb=use_orb,
            full_image=enhanced_image,
        )

        if ground_truth_root is not None:
            ground_truth_label = infer_label_from_path(image_path, ground_truth_root)
        else:
            parent_name = image_path.parent.name.strip().lower()
            ground_truth_label = "tumour" if "tum" in parent_name else "no_tumour" if "no" in parent_name or "normal" in parent_name else None

        folder_name = _gallery_folder(predicted_label, ground_truth_label)
        output_folder = folders[folder_name]
        output_path = output_folder / f"prediction_{index:03d}_{image_path.stem}_pred-{predicted_label}_conf-{confidence:.3f}.png"

        save_case_artifacts(
            original_image,
            enhanced_image,
            segmentation_mask,
            contour_overlay,
            predicted_label,
            ground_truth_label=ground_truth_label,
            confidence=confidence,
            output_path=output_path,
        )
        saved_paths.append(output_path)

    return saved_paths


def save_hog_visualisations(
    sample_paths: list[Path],
    output_dir: str | Path,
    *,
    max_examples: int = 5,
    hog_parameters=None,
    clahe_clip: float = 2.0,
    clahe_grid: tuple[int, int] = (8, 8),
) -> list[Path]:
    """Save HOG image visualisations for a small sample of MRI scans."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_settings = _build_preprocessing_settings(clahe_clip, clahe_grid)
    saved_paths: list[Path] = []

    for index, image_path in enumerate(sample_paths[:max_examples], start=1):
        original_image = load_original_image(image_path)
        enhanced_image = preprocess_image(original_image, preprocessing_settings)
        orientations = getattr(hog_parameters, "orientations", 9)
        pixels_per_cell = getattr(hog_parameters, "pixels_per_cell", (8, 8))
        cells_per_block = getattr(hog_parameters, "cells_per_block", (2, 2))
        hog_features, hog_image = hog(
            enhanced_image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm="L2-Hys",
            visualize=True,
            feature_vector=True,
        )
        del hog_features

        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) if original_image.ndim == 3 else original_image, cmap="gray")
        axes[0].set_title("Original MRI")
        axes[0].axis("off")
        axes[1].imshow(hog_image, cmap="gray")
        axes[1].set_title("HOG representation")
        axes[1].axis("off")
        figure.tight_layout()

        output_path = output_dir / f"hog_example_{index:02d}_{image_path.stem}.png"
        figure.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(output_path)

    return saved_paths