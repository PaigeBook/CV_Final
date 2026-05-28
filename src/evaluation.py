"""Model evaluation and result visualisation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import cv2
from .segmentation import overlay_mask
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
	"""Compute scalar and tabular metrics."""

	return {
		"accuracy": accuracy_score(y_true, y_pred),
		"precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
		"recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
		"f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
		"classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
		"confusion_matrix": confusion_matrix(y_true, y_pred),
	}


def generate_classification_report_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
	"""Return a text classification report."""

	return classification_report(y_true, y_pred, zero_division=0)


def plot_confusion_matrix(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	class_names: list[str],
	*,
	normalize: bool = False,
	title: str = "Confusion Matrix",
	output_path: str | Path | None = None,
) -> plt.Figure:
	"""Plot and optionally save the confusion matrix."""

	matrix = confusion_matrix(y_true, y_pred, labels=class_names)
	if normalize:
		row_sums = matrix.sum(axis=1, keepdims=True)
		row_sums[row_sums == 0] = 1
		matrix = matrix.astype(np.float32) / row_sums

	figure, axis = plt.subplots(figsize=(10, 8))
	image = axis.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
	axis.set_title(title)
	figure.colorbar(image, ax=axis)

	tick_positions = np.arange(len(class_names))
	axis.set_xticks(tick_positions)
	axis.set_yticks(tick_positions)
	axis.set_xticklabels(class_names, rotation=45, ha="right")
	axis.set_yticklabels(class_names)
	axis.set_ylabel("True label")
	axis.set_xlabel("Predicted label")

	threshold = matrix.max() / 2.0 if matrix.size else 0.0
	for row in range(matrix.shape[0]):
		for column in range(matrix.shape[1]):
			value = matrix[row, column]
			axis.text(
				column,
				row,
				f"{value:.2f}" if normalize else f"{int(value)}",
				ha="center",
				va="center",
				color="white" if value > threshold else "black",
			)

	figure.tight_layout()

	if output_path is not None:
		output_path = Path(output_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		figure.savefig(output_path, dpi=300, bbox_inches="tight")

	return figure


def plot_class_distribution(labels: list[str], *, title: str = "Class Distribution", output_path: str | Path | None = None) -> plt.Figure:
	"""Plot class counts."""

	unique_labels, counts = np.unique(np.asarray(labels), return_counts=True)
	figure, axis = plt.subplots(figsize=(10, 5))
	axis.bar(unique_labels, counts, color="#4C78A8")
	axis.set_title(title)
	axis.set_ylabel("Count")
	axis.tick_params(axis="x", rotation=45)
	figure.tight_layout()

	if output_path is not None:
		output_path = Path(output_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		figure.savefig(output_path, dpi=300, bbox_inches="tight")

	return figure


def identify_false_negatives(y_true: np.ndarray, y_pred: np.ndarray, *, positive_label: str = "tumour") -> np.ndarray:
	"""Return the indices where a tumour case was predicted as normal.

	In a medical setting, false negatives are the most critical error class
	because the model misses a tumour that should have been flagged for review.
	"""

	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)
	return np.where((y_true == positive_label) & (y_pred != positive_label))[0]


def save_false_negative_examples(
	image_paths: list[Path],
	y_true: np.ndarray,
	y_pred: np.ndarray,
	output_dir: str | Path,
	*,
	original_images: list[np.ndarray] | None = None,
	masks: list[np.ndarray] | None = None,
	confidences: np.ndarray | None = None,
	positive_label: str = "tumour",
) -> list[Path]:
	"""Save false-negative examples for later medical review."""

	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	false_negative_indices = identify_false_negatives(y_true, y_pred, positive_label=positive_label)
	saved_paths: list[Path] = []

	for index in false_negative_indices:
		image_path = image_paths[int(index)]
		if original_images is not None:
			image = original_images[int(index)]
		else:
			image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
			if image is None:
				continue

		mask = masks[int(index)] if masks is not None and int(index) < len(masks) else None
		annotated = overlay_mask(image, mask) if mask is not None else (cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy())
		if annotated.ndim == 2:
			annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

		banner = annotated.copy()
		confidence_text = "N/A"
		if confidences is not None and int(index) < len(confidences):
			confidence_text = f"{float(confidences[int(index)]):.3f}"
		text_lines = [
			"FALSE NEGATIVE",
			f"GT: {y_true[int(index)]}",
			f"Pred: {y_pred[int(index)]}",
			f"Confidence: {confidence_text}",
		]
		height = 110
		cv2.rectangle(banner, (0, 0), (banner.shape[1], height), (0, 0, 0), thickness=-1)
		for line_no, line in enumerate(text_lines):
			cv2.putText(banner, line, (10, 25 + line_no * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

		output_path = output_dir / f"false_negative_{int(index):03d}_{image_path.stem}.png"
		cv2.imwrite(str(output_path), banner)
		saved_paths.append(output_path)

	return saved_paths


def visualize_prediction(
	image: np.ndarray,
	mask: np.ndarray | None,
	predicted_label: str,
	*,
	output_path: str | Path | None = None,
	overlay_color: tuple[int, int, int] = (255, 0, 0),
	overlay_alpha: float = 0.4,
) -> plt.Figure:
	"""Create a simple visualisation: original image, mask overlay and predicted label banner.

	Returns the Matplotlib figure and optionally saves it to disk.
	"""

	figure, axes = plt.subplots(1, 3, figsize=(15, 6))

	# Original
	axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image, cmap="gray")
	axes[0].set_title("Original")
	axes[0].axis("off")

	# Mask alone
	if mask is None:
		axes[1].text(0.5, 0.5, "No mask", ha="center", va="center")
	else:
		axes[1].imshow(mask, cmap="gray")
	axes[1].set_title("Segmented mask")
	axes[1].axis("off")

	# Overlay
	if mask is None:
		overlay_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
		axes[2].imshow(overlay_img, cmap="gray")
	else:
		color = overlay_color
		overlay_img = overlay_mask(image, mask, color=color, alpha=overlay_alpha)
		axes[2].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
	axes[2].set_title(f"Prediction: {predicted_label}")
	axes[2].axis("off")

	figure.tight_layout()
	if output_path is not None:
		output_path = Path(output_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		figure.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(figure)
	return figure


def format_metric_summary(metrics: dict[str, Any]) -> str:
	"""Format key scalar metrics."""

	return (
		f"Accuracy: {metrics.get('accuracy', 0.0):.4f}\n"
		f"Precision (macro): {metrics.get('precision_macro', 0.0):.4f}\n"
		f"Recall (macro): {metrics.get('recall_macro', 0.0):.4f}\n"
		f"F1-score (macro): {metrics.get('f1_macro', 0.0):.4f}\n"
	)


def format_timing_summary(timing: dict[str, Any]) -> str:
	"""Format timing and efficiency metrics."""

	return (
		f"Preprocessing time: {float(timing.get('preprocessing_time_seconds', 0.0)):.4f} s\n"
		f"Segmentation time: {float(timing.get('segmentation_time_seconds', 0.0)):.4f} s\n"
		f"Feature extraction time: {float(timing.get('feature_extraction_time_seconds', 0.0)):.4f} s\n"
		f"Classification time: {float(timing.get('classification_time_seconds', 0.0)):.4f} s\n"
		f"Total inference time per image: {float(timing.get('total_inference_time_per_image_seconds', 0.0)):.6f} s\n"
		f"SVM training time: {float(timing.get('svm_training_time_seconds', 0.0)):.4f} s\n"
		f"Random Forest training time: {float(timing.get('rf_training_time_seconds', 0.0)):.4f} s\n"
		f"Baseline training time: {float(timing.get('baseline_training_time_seconds', 0.0)):.4f} s\n"
		f"SVM inference time per image: {float(timing.get('svm_inference_time_per_image_seconds', 0.0)):.6f} s\n"
		f"Random Forest inference time per image: {float(timing.get('rf_inference_time_per_image_seconds', 0.0)):.6f} s\n"
		f"Baseline inference time per image: {float(timing.get('baseline_inference_time_per_image_seconds', 0.0)):.6f} s\n"
	)


def build_method_comparison_report(
	hog_svm_metrics: dict[str, Any],
	cnn_metrics: dict[str, Any] | None = None,
	*,
	cnn_note: str = "CNN baseline not trained in this classical pipeline",
) -> str:
	"""Build a HOG+SVM vs CNN comparison report."""

	cnn_metrics = cnn_metrics or {}
	hog_accuracy = float(hog_svm_metrics.get("accuracy", 0.0))
	hog_precision = float(hog_svm_metrics.get("precision_macro", 0.0))
	hog_recall = float(hog_svm_metrics.get("recall_macro", 0.0))
	hog_f1 = float(hog_svm_metrics.get("f1_macro", 0.0))

	cnn_accuracy = cnn_metrics.get("accuracy", "N/A")
	cnn_precision = cnn_metrics.get("precision_macro", "N/A")
	cnn_recall = cnn_metrics.get("recall_macro", "N/A")
	cnn_f1 = cnn_metrics.get("f1_macro", "N/A")

	return (
		"# Method Comparison\n\n"
		"## Quantitative Comparison\n\n"
		"| Method | Accuracy | Precision | Recall | F1-score | Notes |\n"
		"| --- | ---: | ---: | ---: | ---: | --- |\n"
		f"| HOG + SVM | {hog_accuracy:.4f} | {hog_precision:.4f} | {hog_recall:.4f} | {hog_f1:.4f} | Classical baseline implemented in this project |\n"
		f"| CNN-based approach | {cnn_accuracy} | {cnn_precision} | {cnn_recall} | {cnn_f1} | {cnn_note} |\n\n"
		"## Interpretation\n\n"
		"- HOG + SVM is fast, interpretable, and works well as a strong classical baseline for the NEU dataset.\n"
		"- CNN-based approaches usually achieve better accuracy when enough data and compute are available, because they learn features automatically.\n"
		"- The classical pipeline remains transparent because each stage is explicit: preprocessing, segmentation, feature extraction, and classification.\n"
	)


def save_method_comparison_report(
	hog_svm_metrics: dict[str, Any],
	output_path: str | Path,
	cnn_metrics: dict[str, Any] | None = None,
	*,
	cnn_note: str = "CNN baseline not trained in this classical pipeline",
) -> str:
	"""Save and return the comparison report."""

	report_text = build_method_comparison_report(hog_svm_metrics, cnn_metrics=cnn_metrics, cnn_note=cnn_note)
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(report_text, encoding="utf-8")
	return report_text


def build_multi_method_comparison_report(method_metrics: dict[str, dict[str, Any]]) -> str:
	"""Build a comparison report for multiple methods.

	`method_metrics` should map method name -> metrics dict (as returned by `compute_metrics`).
	"""

	lines = ["# Method Comparison\n", "## Quantitative Comparison\n", "| Method | Accuracy | Precision | Recall | F1-score | Notes |\n", "| --- | ---: | ---: | ---: | ---: | --- |\n"]

	for name, metrics in method_metrics.items():
		acc = float(metrics.get("accuracy", 0.0))
		prec = float(metrics.get("precision_macro", 0.0))
		rec = float(metrics.get("recall_macro", 0.0))
		f1 = float(metrics.get("f1_macro", 0.0))
		lines.append(f"| {name} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | |\n")

	lines.append("\n## Interpretation\n\n")
	lines.append("- Results summarize scalar metrics for each method on the same train/test split.\n")
	lines.append("- The raw-pixel baseline provides the minimum expected performance for justification in the report.\n")
	lines.append("- Use confusion matrices and per-class reports to inspect class-specific behaviour.\n")

	return "".join(lines)


def append_timing_section(report_text: str, timing: dict[str, Any] | None) -> str:

	if not timing:
		return report_text

	return (
		report_text
		+ "\n## Efficiency Metrics\n\n"
		+ "| Metric | Value |\n"
		+ "| --- | ---: |\n"
		+ f"| Preprocessing time | {float(timing.get('preprocessing_time_seconds', 0.0)):.4f} s |\n"
		+ f"| Segmentation time | {float(timing.get('segmentation_time_seconds', 0.0)):.4f} s |\n"
		+ f"| Feature extraction time | {float(timing.get('feature_extraction_time_seconds', 0.0)):.4f} s |\n"
		+ f"| Classification time | {float(timing.get('classification_time_seconds', 0.0)):.4f} s |\n"
		+ f"| Total inference time per image | {float(timing.get('total_inference_time_per_image_seconds', 0.0)):.6f} s |\n"
		+ f"| SVM training time | {float(timing.get('svm_training_time_seconds', 0.0)):.4f} s |\n"
		+ f"| Random Forest training time | {float(timing.get('rf_training_time_seconds', 0.0)):.4f} s |\n"
		+ f"| Baseline training time | {float(timing.get('baseline_training_time_seconds', 0.0)):.4f} s |\n"
		+ f"| SVM inference time per image | {float(timing.get('svm_inference_time_per_image_seconds', 0.0)):.6f} s |\n"
		+ f"| Random Forest inference time per image | {float(timing.get('rf_inference_time_per_image_seconds', 0.0)):.6f} s |\n"
		+ f"| Baseline inference time per image | {float(timing.get('baseline_inference_time_per_image_seconds', 0.0)):.6f} s |\n"
	)


def build_efficiency_report(timing: dict[str, Any]) -> str:

	return (
		"# Efficiency Metrics\n\n"
		"## Timing Summary\n\n"
		f"- Preprocessing time: {float(timing.get('preprocessing_time_seconds', 0.0)):.4f} s\n"
		f"- Segmentation time: {float(timing.get('segmentation_time_seconds', 0.0)):.4f} s\n"
		f"- Feature extraction time: {float(timing.get('feature_extraction_time_seconds', 0.0)):.4f} s\n"
		f"- Classification time: {float(timing.get('classification_time_seconds', 0.0)):.4f} s\n"
		f"- Total inference time per image: {float(timing.get('total_inference_time_per_image_seconds', 0.0)):.6f} s\n"
		f"- SVM training time: {float(timing.get('svm_training_time_seconds', 0.0)):.4f} s\n"
		f"- Random Forest training time: {float(timing.get('rf_training_time_seconds', 0.0)):.4f} s\n"
		f"- Baseline training time: {float(timing.get('baseline_training_time_seconds', 0.0)):.4f} s\n"
		f"- SVM inference time per image: {float(timing.get('svm_inference_time_per_image_seconds', 0.0)):.6f} s\n"
		f"- Random Forest inference time per image: {float(timing.get('rf_inference_time_per_image_seconds', 0.0)):.6f} s\n"
		f"- Baseline inference time per image: {float(timing.get('baseline_inference_time_per_image_seconds', 0.0)):.6f} s\n"
	)


def save_efficiency_report(timing: dict[str, Any], output_path: str | Path) -> str:

	report_text = build_efficiency_report(timing)
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(report_text, encoding="utf-8")
	return report_text


def save_multi_method_comparison_report(method_metrics: dict[str, dict[str, Any]], output_path: str | Path) -> str:

	report_text = build_multi_method_comparison_report(method_metrics)
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(report_text, encoding="utf-8")
	return report_text


def save_multi_method_comparison_report_with_timing(
	method_metrics: dict[str, dict[str, Any]],
	output_path: str | Path,
	timing: dict[str, Any] | None = None,
) -> str:

	report_text = append_timing_section(build_multi_method_comparison_report(method_metrics), timing)
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(report_text, encoding="utf-8")
	return report_text
