"""End-to-end orchestration for the classical computer vision pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import ProjectConfig
from src.data_loader import collect_image_records, ensure_dataset_available
from src.features import extract_hog_features
from src.model import build_classifier, evaluate_classifier, split_dataset
from src.preprocessing import load_original_image, preprocess_image
from src.segmentation import segment_suspicious_region
from src.visualization import (
    ensure_output_structure,
    save_case_artifacts,
    save_case_comparison_grid,
    save_confusion_matrix,
)


def _prepare_feature_vector(image_path: Path, config: ProjectConfig) -> tuple[np.ndarray, dict[str, object]]:
    original_image = load_original_image(image_path)
    enhanced_image = preprocess_image(original_image, config)
    segmentation_mask, contour_overlay, suspicious_mask = segment_suspicious_region(
        enhanced_image,
        config.morph_kernel_size,
    )
    masked_image = enhanced_image.copy()
    masked_image[suspicious_mask == 0] = 0
    hog_features = extract_hog_features(
        masked_image,
        orientations=config.hog_orientations,
        pixels_per_cell=config.hog_pixels_per_cell,
        cells_per_block=config.hog_cells_per_block,
    )
    preview = {
        "original_image": original_image,
        "enhanced_image": enhanced_image,
        "segmentation_mask": segmentation_mask,
        "contour_overlay": contour_overlay,
    }
    return hog_features, preview


def run_pipeline(config: ProjectConfig, sample_limit: int = 4) -> dict[str, object]:
    dataset_root = ensure_dataset_available(config.data_root, config.zip_path)
    output_paths = ensure_output_structure(config.output_dir)

    records = collect_image_records(dataset_root)
    features: list[np.ndarray] = []
    labels: list[int] = []
    previews: list[tuple[Path, dict[str, object], int, np.ndarray]] = []

    for record in records:
        feature_vector, preview = _prepare_feature_vector(record.image_path, config)
        features.append(feature_vector)
        labels.append(record.label)
        previews.append((record.image_path, preview, record.label, feature_vector))

    feature_matrix = np.vstack(features)
    label_array = np.asarray(labels, dtype=np.int32)

    split = split_dataset(
        feature_matrix,
        label_array,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    model = build_classifier()
    model.fit(split.x_train, split.y_train)

    metrics = evaluate_classifier(model, split.x_test, split.y_test)
    save_confusion_matrix(metrics["confusion_matrix"], output_paths["reports"] / "comparisons" / "confusion_matrix.png")

    sample_count = min(sample_limit, len(previews))
    selected_indices = np.random.default_rng(config.random_state).choice(
        len(previews),
        size=sample_count,
        replace=False,
    )

    comparison_items: list[tuple[np.ndarray, str, str]] = []

    for index, preview_index in enumerate(selected_indices, start=1):
        image_path, preview, label, feature_vector = previews[int(preview_index)]
        actual_label = "tumour" if label == 1 else "no_tumour"
        predicted_label = "tumour" if model.predict(feature_vector.reshape(1, -1))[0] == 1 else "no_tumour"
        comparison_items.append((preview["original_image"], actual_label, predicted_label))
        save_case_artifacts(
            original_image=preview["original_image"],
            enhanced_image=preview["enhanced_image"],
            segmentation_mask=preview["segmentation_mask"],
            contour_overlay=preview["contour_overlay"],
            predicted_label=predicted_label,
            output_path=output_paths["cases"] / f"case_{index:02d}_{image_path.stem}.png",
        )

    save_case_comparison_grid(
        comparison_items,
        output_paths["reports"] / "comparisons" / "random_case_comparison.png",
    )

    report_path = output_paths["reports"] / "metrics.txt"
    report_path.write_text(
        "\n".join(
            [
                f"accuracy: {metrics['accuracy']}",
                f"precision: {metrics['precision']}",
                f"recall: {metrics['recall']}",
                f"f1_score: {metrics['f1_score']}",
                f"confusion_matrix: {metrics['confusion_matrix']}",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "dataset_root": str(dataset_root),
        "output_dir": str(config.output_dir),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "confusion_matrix": metrics["confusion_matrix"],
    }
