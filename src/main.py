"""Pipeline entry point for Brain MRI tumour detection (classical CV pipeline).

This script orchestrates preprocessing, segmentation, feature extraction and
classification using HOG + SVM (optionally with texture features). Outputs
are saved under the `outputs/` folder (relative paths).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
import zipfile
import os
import shutil
import tempfile
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
	from .classifier import (
		build_baseline_classifier,
		build_svm_classifier,
		build_random_forest_classifier,
		load_dataset,
		infer_label_from_path,
		predict_image,
		predict_features_with_confidence,
		predict_features,
		prepare_feature_matrix,
		prepare_raw_pixel_matrix,
		save_model,
		split_dataset,
		train_classifier,
		PreprocessingSettings,
		SVMConfig,
	)
	from .evaluation import (
		compute_metrics,
		format_metric_summary,
		format_timing_summary,
		generate_classification_report_text,
		save_false_negative_examples,
		save_efficiency_report,
		save_method_comparison_report,
		save_multi_method_comparison_report,
		save_multi_method_comparison_report_with_timing,
		plot_class_distribution,
		plot_confusion_matrix,
	)
	from .feature_extraction import HOGParameters
	from .preprocessing import preprocess_image, load_original_image
	from .segmentation import annotate_defects, segment_defects, overlay_mask
	from .evaluation import visualize_prediction
	from .utils import prepare_dataset_directory, select_diverse_sample_paths
	from .viz import overlay_label, save_visual_summary, save_annotated_predictions, save_hog_visualisations
except ImportError:  # pragma: no cover - fallback for direct script execution
	project_root = Path(__file__).resolve().parents[1]
	if str(project_root) not in sys.path:
		sys.path.insert(0, str(project_root))
	from src.classifier import (  # type: ignore
		build_baseline_classifier,
		build_svm_classifier,
		build_random_forest_classifier,
		load_dataset,
		infer_label_from_path,
		predict_image,
		predict_features_with_confidence,
		predict_features,
		prepare_feature_matrix,
		prepare_raw_pixel_matrix,
		save_model,
		split_dataset,
		train_classifier,
		PreprocessingSettings,
		SVMConfig,
	)
	from src.evaluation import (  # type: ignore
		compute_metrics,
		format_metric_summary,
		format_timing_summary,
		generate_classification_report_text,
		save_false_negative_examples,
		save_efficiency_report,
		save_method_comparison_report,
		save_multi_method_comparison_report,
		save_multi_method_comparison_report_with_timing,
		plot_class_distribution,
		plot_confusion_matrix,
	)
	from src.feature_extraction import HOGParameters  # type: ignore
	from src.preprocessing import preprocess_image, load_original_image  # type: ignore
	from src.segmentation import annotate_defects, segment_defects, overlay_mask  # type: ignore
	from src.evaluation import visualize_prediction  # type: ignore
	from src.utils import prepare_dataset_directory, select_diverse_sample_paths  # type: ignore
	from src.viz import overlay_label, save_visual_summary, save_annotated_predictions, save_hog_visualisations  # type: ignore


def build_argument_parser() -> argparse.ArgumentParser:
	"""Build CLI arguments."""

	parser = argparse.ArgumentParser(
		description="Brain MRI Tumour Detection and Classification using Classical CV Techniques",
	)
	parser.add_argument("--dataset-dir", type=Path, default=Path("data/raw/mri"), help="Path to the MRI dataset (subfolders tumour/ no_tumour/)")
	parser.add_argument("--sample-dir", type=Path, default=None, help="Optional folder of sample images for annotation")
	parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for evaluation and prediction outputs")
	parser.add_argument("--model-path", type=Path, default=Path("models/mri_svm.joblib"), help="Path where the trained model is saved")
	parser.add_argument("--rf-model-path", type=Path, default=Path("models/mri_rf.joblib"), help="Path where the Random Forest model is saved")
	parser.add_argument("--baseline-model-path", type=Path, default=Path("models/mri_baseline.joblib"), help="Path where the raw-pixel baseline model is saved")
	parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of the dataset reserved for testing")
	parser.add_argument("--hog-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(224, 224), help="Resize images before HOG extraction")
	parser.add_argument("--hog-orientations", type=int, default=9, help="Number of HOG orientation bins")
	parser.add_argument("--hog-pixels-per-cell", type=int, nargs=2, metavar=("X", "Y"), default=(8, 8), help="HOG pixels per cell")
	parser.add_argument("--hog-cells-per-block", type=int, nargs=2, metavar=("X", "Y"), default=(2, 2), help="HOG cells per block")
	parser.add_argument("--baseline-image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(128, 128), help="Resize size for the raw-pixel baseline model")
	parser.add_argument("--svm-kernel", type=str, default="rbf", help="SVM kernel type")
	parser.add_argument("--svm-c", type=float, default=5.0, help="SVM regularisation strength")
	parser.add_argument("--morph-kernel-size", type=int, default=5, help="Morphological kernel size used during ROI segmentation")
	parser.add_argument("--skip-baseline", action="store_true", help="Skip training the raw-pixel baseline model")
	parser.add_argument("--segmentation-method", type=str, default="mri_enhanced", choices=["edge", "hybrid", "watershed", "mri_enhanced"], help="Segmentation approach for tumour candidates")
	parser.add_argument("--num-sample-predictions", type=int, default=5, help="Number of annotated sample images to save")
	parser.add_argument("--use-glcm", action="store_true", help="Include GLCM texture features alongside HOG")
	parser.add_argument("--use-orb", action="store_true", help="Include ORB descriptor summary alongside HOG")
	parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit for contrast enhancement")
	parser.add_argument("--clahe-grid", type=int, nargs=2, metavar=("X", "Y"), default=(8, 8), help="CLAHE tile grid size")
	return parser


def ensure_output_directories(output_dir: Path) -> dict[str, Path]:
	"""Create output folders."""

	paths = {
		"root": output_dir,
		"figures": output_dir / "figures",
		"predictions": output_dir / "predictions",
		"gallery": output_dir / "gallery",
		"features": output_dir / "features",
		"misclassified": output_dir / "misclassified",
		"false_negatives": output_dir / "misclassified" / "false_negatives",
		"reports": output_dir / "reports",
		"comparisons": output_dir / "reports" / "comparisons",
		"models": output_dir / "models",
		"logs": output_dir / "logs",
	}
	for path in paths.values():
		path.mkdir(parents=True, exist_ok=True)
	return paths


# prepare_dataset_directory moved to src.utils


# overlay_label moved to src.viz


def build_hog_parameters(args: argparse.Namespace) -> HOGParameters:
	"""Build HOG parameters from CLI args."""

	return HOGParameters(
		resize_to=(int(args.hog_size[0]), int(args.hog_size[1])),
		orientations=int(args.hog_orientations),
		pixels_per_cell=(int(args.hog_pixels_per_cell[0]), int(args.hog_pixels_per_cell[1])),
		cells_per_block=(int(args.hog_cells_per_block[0]), int(args.hog_cells_per_block[1])),
	)


def build_svm_config(args: argparse.Namespace, hog_parameters: HOGParameters) -> SVMConfig:
	"""Build SVM config from CLI args."""

	return SVMConfig(kernel=args.svm_kernel, c=args.svm_c, hog_parameters=hog_parameters)


def build_preprocessing_settings(args: argparse.Namespace) -> PreprocessingSettings:
	"""Build the preprocessing settings used throughout the pipeline."""

	return PreprocessingSettings(
		image_size=(int(args.hog_size[0]), int(args.hog_size[1])),
		gaussian_kernel=(5, 5),
		clahe_clip_limit=float(args.clahe_clip),
		clahe_tile_grid_size=(int(args.clahe_grid[0]), int(args.clahe_grid[1])),
	)


# select_diverse_sample_paths moved to src.utils


# save_visual_summary moved to src.viz


# save_annotated_predictions moved to src.viz


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
	"""Run training, evaluation, and output generation."""

	output_dirs = ensure_output_directories(args.output_dir)
	project_root = Path(__file__).resolve().parents[1]
	dataset_dir = prepare_dataset_directory(args.dataset_dir, project_root)

	# Quick sanity check: ensure dataset contains images (or zip was available)
	image_extensions = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
	found_images = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_extensions]
	if not found_images:
		print(f"[ERROR] No images found under '{dataset_dir}'.\nPlace images under '{dataset_dir}/tumour' and '{dataset_dir}/no_tumour', or put 'brain_tumor_dataset.zip' in the repository root and re-run.")
		raise SystemExit(1)
	hog_parameters = build_hog_parameters(args)
	svm_config = build_svm_config(args, hog_parameters)
	preprocessing_settings = build_preprocessing_settings(args)

	dataset = load_dataset(dataset_dir, preprocess=False)
	plot_class_distribution(dataset.labels, output_path=output_dirs["figures"] / "class_distribution.png")

	preprocessed_images: list[np.ndarray] = []
	hog_ready_images: list[np.ndarray] = []
	raw_pixel_images: list[np.ndarray] = []
	segmentation_masks: list[np.ndarray] = []
	contour_overlays: list[np.ndarray] = []
	suspicious_masks: list[np.ndarray] = []
	preprocessing_time_seconds = 0.0
	segmentation_time_seconds = 0.0
	feature_extraction_time_seconds = 0.0
	baseline_feature_extraction_time_seconds = 0.0
	svm_training_time_seconds = 0.0
	rf_training_time_seconds = 0.0
	baseline_training_time_seconds = 0.0
	svm_inference_time_per_image_seconds = 0.0
	rf_inference_time_per_image_seconds = 0.0
	baseline_inference_time_per_image_seconds = 0.0
	classification_time_seconds = 0.0
	total_inference_time_per_image_seconds = 0.0

	for image in dataset.images:
		preprocess_start = perf_counter()
		enhanced_image = preprocess_image(image, preprocessing_settings)
		preprocessing_time_seconds += perf_counter() - preprocess_start
		preprocessed_images.append(enhanced_image)

		segmentation_start = perf_counter()
		segmentation_mask, contour_overlay, suspicious_mask = segment_defects(enhanced_image, args.morph_kernel_size)
		segmentation_time_seconds += perf_counter() - segmentation_start
		segmentation_masks.append(segmentation_mask)
		contour_overlays.append(contour_overlay)
		suspicious_masks.append(suspicious_mask)

		masked_image = enhanced_image.copy()
		masked_image[suspicious_mask == 0] = 0
		hog_ready_images.append(masked_image)
		raw_pixel_images.append(image)

	feature_start = perf_counter()
	hog_features = prepare_feature_matrix(hog_ready_images, hog_parameters=hog_parameters, use_glcm=args.use_glcm, use_orb=args.use_orb)
	feature_extraction_time_seconds = perf_counter() - feature_start

	raw_feature_start = perf_counter()
	raw_pixel_features = prepare_raw_pixel_matrix(raw_pixel_images, resize_to=(int(args.baseline_image_size[0]), int(args.baseline_image_size[1])))
	baseline_feature_extraction_time_seconds = perf_counter() - raw_feature_start

	split = split_dataset(hog_features, dataset.labels, test_size=args.test_size, random_state=42)
	train_indices = split.train_indices
	test_indices = split.test_indices
	raw_x_test = raw_pixel_features[test_indices]
	image_paths_test = [dataset.file_paths[int(index)] for index in test_indices]
	original_test_images = [dataset.images[int(index)] for index in test_indices]
	test_masks = [suspicious_masks[int(index)] for index in test_indices]
	preprocessed_test_images = [preprocessed_images[int(index)] for index in test_indices]

	model = build_svm_classifier(svm_config)
	try:
		svm_train_start = perf_counter()
		train_classifier(model, split.x_train, split.y_train)
		svm_training_time_seconds = perf_counter() - svm_train_start
	except Exception as exc:
		print(f"[ERROR] SVM training failed: {exc}")
		svm_training_time_seconds = 0.0

	# Evaluate the HOG + SVM model on the shared test split.
	try:
		svm_inference_start = perf_counter()
		svm_predictions, svm_confidences = predict_features_with_confidence(model, split.x_test)
		svm_inference_total = perf_counter() - svm_inference_start
		svm_inference_time_per_image_seconds = svm_inference_total / max(len(split.x_test), 1)
		classification_time_seconds = svm_inference_time_per_image_seconds
		svm_metrics = compute_metrics(split.y_test, svm_predictions)
	except Exception as exc:
		print(f"[ERROR] SVM evaluation failed: {exc}")
		svm_predictions = np.array([])
		svm_confidences = np.array([])
		svm_metrics = {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}

	# Train and evaluate Random Forest as an alternative classifier.
	rf_metrics = {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}
	try:
		rf_model = build_random_forest_classifier()
		rf_train_start = perf_counter()
		train_classifier(rf_model, split.x_train, split.y_train)
		rf_training_time_seconds = perf_counter() - rf_train_start
		rf_inference_start = perf_counter()
		rf_y_pred = predict_features(rf_model, split.x_test)
		rf_inference_total = perf_counter() - rf_inference_start
		rf_inference_time_per_image_seconds = rf_inference_total / max(len(split.x_test), 1)
		rf_metrics = compute_metrics(split.y_test, rf_y_pred)
	except Exception as exc:
		print(f"[WARN] Random Forest training/evaluation failed: {exc}")

	baseline_metrics: dict[str, object] | None = None
	baseline_model = None
	baseline_y_pred: np.ndarray | None = None
	if not args.skip_baseline:
		try:
			baseline_model = build_baseline_classifier()
			baseline_train_start = perf_counter()
			train_classifier(baseline_model, raw_pixel_features[train_indices], split.y_train)
			baseline_training_time_seconds = perf_counter() - baseline_train_start
			baseline_inference_start = perf_counter()
			baseline_y_pred = predict_features(baseline_model, raw_x_test)
			baseline_inference_total = perf_counter() - baseline_inference_start
			baseline_inference_time_per_image_seconds = baseline_inference_total / max(len(split.x_test), 1)
			baseline_metrics = compute_metrics(split.y_test, baseline_y_pred)
		except Exception as exc:
			print(f"[WARN] Baseline training/evaluation failed: {exc}")
			baseline_metrics = {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}

	# Save confusion matrices for all methods using the same class order.
	present = set(dataset.labels.tolist())
	class_names = [c for c in ("no_tumour", "tumour") if c in present]
	figure = plot_confusion_matrix(split.y_test, svm_predictions, class_names, output_path=output_dirs["figures"] / "confusion_matrix_svm.png")
	plt.close(figure)
	figure = plot_confusion_matrix(split.y_test, rf_y_pred, class_names, output_path=output_dirs["figures"] / "confusion_matrix_rf.png")
	plt.close(figure)
	if baseline_y_pred is not None:
		figure = plot_confusion_matrix(split.y_test, baseline_y_pred, class_names, output_path=output_dirs["figures"] / "confusion_matrix_baseline.png")
		plt.close(figure)

	# Save classification reports and scalar metrics.
	(report_dir := output_dirs["reports"]).mkdir(parents=True, exist_ok=True)
	(report_dir / "classification_report_svm.txt").write_text(generate_classification_report_text(split.y_test, svm_predictions), encoding="utf-8")
	(report_dir / "classification_report_rf.txt").write_text(generate_classification_report_text(split.y_test, rf_y_pred), encoding="utf-8")
	if baseline_y_pred is not None:
		(report_dir / "classification_report_baseline.txt").write_text(generate_classification_report_text(split.y_test, baseline_y_pred), encoding="utf-8")
	(report_dir / "metrics_svm.txt").write_text(format_metric_summary(svm_metrics), encoding="utf-8")
	(report_dir / "metrics_rf.txt").write_text(format_metric_summary(rf_metrics), encoding="utf-8")
	if baseline_metrics is not None:
		(report_dir / "metrics_baseline.txt").write_text(format_metric_summary(baseline_metrics), encoding="utf-8")
	(report_dir / "timing.txt").write_text(
		format_timing_summary(
			{
				"preprocessing_time_seconds": preprocessing_time_seconds / max(len(dataset.images), 1),
				"segmentation_time_seconds": segmentation_time_seconds / max(len(dataset.images), 1),
				"feature_extraction_time_seconds": feature_extraction_time_seconds / max(len(dataset.images), 1),
				"classification_time_seconds": classification_time_seconds,
				"total_inference_time_per_image_seconds": (preprocessing_time_seconds + segmentation_time_seconds + feature_extraction_time_seconds) / max(len(dataset.images), 1) + classification_time_seconds,
				"svm_training_time_seconds": svm_training_time_seconds,
				"rf_training_time_seconds": rf_training_time_seconds,
				"baseline_training_time_seconds": baseline_training_time_seconds,
				"svm_inference_time_per_image_seconds": svm_inference_time_per_image_seconds,
				"rf_inference_time_per_image_seconds": rf_inference_time_per_image_seconds,
				"baseline_inference_time_per_image_seconds": baseline_inference_time_per_image_seconds,
			}
		),
		encoding="utf-8",
	)

	# Save a multi-method comparison report.
	method_metrics: dict[str, dict[str, object]] = {"HOG + SVM": svm_metrics, "HOG + Random Forest": rf_metrics}
	if baseline_metrics is not None:
		method_metrics["Raw Pixels + Logistic Regression"] = baseline_metrics
	save_multi_method_comparison_report_with_timing(
		method_metrics,
		output_dirs["comparisons"] / "method_comparison.md",
		{
			"preprocessing_time_seconds": preprocessing_time_seconds / max(len(dataset.images), 1),
			"segmentation_time_seconds": segmentation_time_seconds / max(len(dataset.images), 1),
			"feature_extraction_time_seconds": feature_extraction_time_seconds / max(len(dataset.images), 1),
			"classification_time_seconds": classification_time_seconds,
			"total_inference_time_per_image_seconds": (preprocessing_time_seconds + segmentation_time_seconds + feature_extraction_time_seconds) / max(len(dataset.images), 1) + classification_time_seconds,
			"svm_training_time_seconds": svm_training_time_seconds,
			"rf_training_time_seconds": rf_training_time_seconds,
			"baseline_training_time_seconds": baseline_training_time_seconds,
			"svm_inference_time_per_image_seconds": svm_inference_time_per_image_seconds,
			"rf_inference_time_per_image_seconds": rf_inference_time_per_image_seconds,
			"baseline_inference_time_per_image_seconds": baseline_inference_time_per_image_seconds,
		},
	)
	save_efficiency_report(
		{
			"preprocessing_time_seconds": preprocessing_time_seconds / max(len(dataset.images), 1),
			"segmentation_time_seconds": segmentation_time_seconds / max(len(dataset.images), 1),
			"feature_extraction_time_seconds": feature_extraction_time_seconds / max(len(dataset.images), 1),
			"classification_time_seconds": classification_time_seconds,
			"total_inference_time_per_image_seconds": (preprocessing_time_seconds + segmentation_time_seconds + feature_extraction_time_seconds) / max(len(dataset.images), 1) + classification_time_seconds,
			"svm_training_time_seconds": svm_training_time_seconds,
			"rf_training_time_seconds": rf_training_time_seconds,
			"baseline_training_time_seconds": baseline_training_time_seconds,
			"svm_inference_time_per_image_seconds": svm_inference_time_per_image_seconds,
			"rf_inference_time_per_image_seconds": rf_inference_time_per_image_seconds,
			"baseline_inference_time_per_image_seconds": baseline_inference_time_per_image_seconds,
		},
		output_dirs["comparisons"] / "efficiency_metrics.md",
	)

	# Persist models.
	save_model(model, args.model_path)
	save_model(rf_model, args.rf_model_path)
	if baseline_model is not None:
		save_model(baseline_model, args.baseline_model_path)

	# Run a compact per-image inference timing pass on the SVM model.
	# This keeps the timing numbers tied to the medical-review output that is
	# also used for false-negative analysis.
	if len(test_indices) > 0:
		timing_preprocess = 0.0
		timing_segment = 0.0
		timing_features = 0.0
		timing_classify = 0.0
		for image_path in image_paths_test:
			original_image = load_original_image(image_path)
			stage_start = perf_counter()
			enhanced_image = preprocess_image(original_image, preprocessing_settings)
			timing_preprocess += perf_counter() - stage_start

			stage_start = perf_counter()
			segmentation_mask, _, suspicious_mask = segment_defects(enhanced_image, args.morph_kernel_size)
			timing_segment += perf_counter() - stage_start

			masked_image = enhanced_image.copy()
			masked_image[suspicious_mask == 0] = 0

			stage_start = perf_counter()
			feature_vector = prepare_feature_matrix([masked_image], hog_parameters=hog_parameters, use_glcm=args.use_glcm, use_orb=args.use_orb)
			timing_features += perf_counter() - stage_start

			stage_start = perf_counter()
			predict_features_with_confidence(model, feature_vector)
			timing_classify += perf_counter() - stage_start

		preprocessing_time_seconds = timing_preprocess
		segmentation_time_seconds = timing_segment
		feature_extraction_time_seconds = timing_features
		classification_time_seconds = timing_classify
		total_inference_time_per_image_seconds = (
			preprocessing_time_seconds + segmentation_time_seconds + feature_extraction_time_seconds + classification_time_seconds
		) / max(len(test_indices), 1)

	# False negative analysis on the SVM model.
	false_negative_paths = save_false_negative_examples(
		image_paths_test,
		split.y_test,
		svm_predictions,
		output_dirs["false_negatives"],
		original_images=original_test_images,
		masks=test_masks,
		confidences=svm_confidences,
	)
	print(f"False negatives found: {len(false_negative_paths)}")

	# Save gallery and HOG visualisations.
	gallery_paths = save_annotated_predictions(
		model,
		image_paths_test,
		output_dirs["gallery"],
		output_dirs["figures"],
		hog_parameters=hog_parameters,
		segmentation_method=args.segmentation_method,
		use_glcm=args.use_glcm,
		use_orb=args.use_orb,
		use_clahe=True,
		clahe_clip=float(args.clahe_clip),
		clahe_grid=(int(args.clahe_grid[0]), int(args.clahe_grid[1])),
		morph_kernel_size=args.morph_kernel_size,
		ground_truth_root=dataset_dir,
	)
	feature_visual_paths = save_hog_visualisations(
		image_paths_test,
		output_dirs["features"],
		max_examples=min(3, max(1, int(args.num_sample_predictions))),
		hog_parameters=hog_parameters,
		clahe_clip=float(args.clahe_clip),
		clahe_grid=(int(args.clahe_grid[0]), int(args.clahe_grid[1])),
	)

	timing_summary = {
		"preprocessing_time_seconds": preprocessing_time_seconds / max(len(dataset.images), 1),
		"segmentation_time_seconds": segmentation_time_seconds / max(len(dataset.images), 1),
		"feature_extraction_time_seconds": feature_extraction_time_seconds / max(len(dataset.images), 1),
		"classification_time_seconds": classification_time_seconds,
		"total_inference_time_per_image_seconds": total_inference_time_per_image_seconds,
		"svm_training_time_seconds": svm_training_time_seconds,
		"rf_training_time_seconds": rf_training_time_seconds,
		"baseline_training_time_seconds": baseline_training_time_seconds,
		"svm_inference_time_per_image_seconds": svm_inference_time_per_image_seconds,
		"rf_inference_time_per_image_seconds": rf_inference_time_per_image_seconds,
		"baseline_inference_time_per_image_seconds": baseline_inference_time_per_image_seconds,
	}
	(report_dir / "timing.txt").write_text(format_timing_summary(timing_summary), encoding="utf-8")
	save_multi_method_comparison_report_with_timing(method_metrics, output_dirs["comparisons"] / "method_comparison.md", timing_summary)
	save_efficiency_report(timing_summary, output_dirs["comparisons"] / "efficiency_metrics.md")

	return {
		"svm_metrics": svm_metrics,
		"rf_metrics": rf_metrics,
		"baseline_metrics": baseline_metrics,
		"timing": timing_summary,
		"svm_model_path": args.model_path,
		"rf_model_path": args.rf_model_path,
		"baseline_model_path": args.baseline_model_path,
		"gallery_paths": gallery_paths,
		"feature_visual_paths": feature_visual_paths,
		"false_negative_paths": false_negative_paths,
	}


def main() -> int:
	"""CLI entry point."""

	parser = build_argument_parser()
	args = parser.parse_args()
	results = run_pipeline(args)

	print("SVM results:")
	print(f"  Accuracy: {results['svm_metrics']['accuracy']:.4f}")
	print(f"  Precision (macro): {results['svm_metrics']['precision_macro']:.4f}")
	print(f"  Recall (macro): {results['svm_metrics']['recall_macro']:.4f}")
	print(f"  F1-score (macro): {results['svm_metrics']['f1_macro']:.4f}")
	print(f"  Model saved to: {results['svm_model_path']}")

	print("Random Forest results:")
	print(f"  Accuracy: {results['rf_metrics']['accuracy']:.4f}")
	print(f"  Precision (macro): {results['rf_metrics']['precision_macro']:.4f}")
	print(f"  Recall (macro): {results['rf_metrics']['recall_macro']:.4f}")
	print(f"  F1-score (macro): {results['rf_metrics']['f1_macro']:.4f}")
	print(f"  Model saved to: {results['rf_model_path']}")
	if results.get("baseline_metrics") is not None:
		print("Raw-pixel baseline results:")
		print(f"  Accuracy: {results['baseline_metrics']['accuracy']:.4f}")
		print(f"  Precision (macro): {results['baseline_metrics']['precision_macro']:.4f}")
		print(f"  Recall (macro): {results['baseline_metrics']['recall_macro']:.4f}")
		print(f"  F1-score (macro): {results['baseline_metrics']['f1_macro']:.4f}")
		print(f"  Model saved to: {results['baseline_model_path']}")

	print("Timing summary:")
	print(f"  Preprocessing time: {results['timing']['preprocessing_time_seconds']:.4f} s")
	print(f"  Segmentation time: {results['timing']['segmentation_time_seconds']:.4f} s")
	print(f"  Feature extraction time: {results['timing']['feature_extraction_time_seconds']:.4f} s")
	print(f"  Classification time: {results['timing']['classification_time_seconds']:.4f} s")
	print(f"  Total inference time / image: {results['timing']['total_inference_time_per_image_seconds']:.6f} s")
	print(f"  SVM training time: {results['timing']['svm_training_time_seconds']:.4f} s")
	print(f"  RF training time: {results['timing']['rf_training_time_seconds']:.4f} s")
	print(f"  Baseline training time: {results['timing']['baseline_training_time_seconds']:.4f} s")
	print(f"  SVM inference time / image: {results['timing']['svm_inference_time_per_image_seconds']:.6f} s")
	print(f"  RF inference time / image: {results['timing']['rf_inference_time_per_image_seconds']:.6f} s")
	print(f"  Baseline inference time / image: {results['timing']['baseline_inference_time_per_image_seconds']:.6f} s")
	print(f"False-negative examples saved: {len(results['false_negative_paths'])}")
	print(f"Gallery images saved: {len(results['gallery_paths'])}")
	print(f"HOG feature visualisations saved: {len(results['feature_visual_paths'])}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
