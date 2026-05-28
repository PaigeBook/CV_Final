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
		build_svm_classifier,
		build_random_forest_classifier,
		load_dataset,
		infer_label_from_path,
		predict_image,
		predict_features,
		prepare_feature_matrix,
		save_model,
		split_dataset,
		train_classifier,
		SVMConfig,
	)
	from .evaluation import (
		compute_metrics,
		format_metric_summary,
		format_timing_summary,
		generate_classification_report_text,
		save_efficiency_report,
		save_method_comparison_report,
		save_multi_method_comparison_report,
		save_multi_method_comparison_report_with_timing,
		plot_class_distribution,
		plot_confusion_matrix,
	)
	from .feature_extraction import HOGParameters
	from .preprocessing import preprocess_image, read_image
	from .segmentation import annotate_defects, segment_defects, overlay_mask
	from .evaluation import visualize_prediction
	from .utils import prepare_dataset_directory, select_diverse_sample_paths
	from .viz import overlay_label, save_visual_summary, save_annotated_predictions
except ImportError:  # pragma: no cover - fallback for direct script execution
	project_root = Path(__file__).resolve().parents[1]
	if str(project_root) not in sys.path:
		sys.path.insert(0, str(project_root))
	from src.classifier import (  # type: ignore
		build_svm_classifier,
		build_random_forest_classifier,
		load_dataset,
		infer_label_from_path,
		predict_image,
		predict_features,
		prepare_feature_matrix,
		save_model,
		split_dataset,
		train_classifier,
		SVMConfig,
	)
	from src.evaluation import (  # type: ignore
		compute_metrics,
		format_metric_summary,
		format_timing_summary,
		generate_classification_report_text,
		save_efficiency_report,
		save_method_comparison_report,
		save_multi_method_comparison_report,
		save_multi_method_comparison_report_with_timing,
		plot_class_distribution,
		plot_confusion_matrix,
	)
	from src.feature_extraction import HOGParameters  # type: ignore
	from src.preprocessing import preprocess_image, read_image  # type: ignore
	from src.segmentation import annotate_defects, segment_defects, overlay_mask  # type: ignore
	from src.evaluation import visualize_prediction  # type: ignore
	from src.utils import prepare_dataset_directory, select_diverse_sample_paths  # type: ignore
	from src.viz import overlay_label, save_visual_summary, save_annotated_predictions  # type: ignore


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
	parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of the dataset reserved for testing")
	parser.add_argument("--hog-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(224, 224), help="Resize images before HOG extraction")
	parser.add_argument("--hog-orientations", type=int, default=9, help="Number of HOG orientation bins")
	parser.add_argument("--hog-pixels-per-cell", type=int, nargs=2, metavar=("X", "Y"), default=(8, 8), help="HOG pixels per cell")
	parser.add_argument("--hog-cells-per-block", type=int, nargs=2, metavar=("X", "Y"), default=(2, 2), help="HOG cells per block")
	parser.add_argument("--svm-kernel", type=str, default="rbf", help="SVM kernel type")
	parser.add_argument("--svm-c", type=float, default=5.0, help="SVM regularisation strength")
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
		"reports": output_dir / "reports",
		"comparisons": output_dir / "reports" / "comparisons",
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

	dataset = load_dataset(
		dataset_dir,
		preprocess=True,
		target_size=hog_parameters.resize_to,
		use_clahe=True,
		clahe_clip_limit=float(args.clahe_clip),
		clahe_tile_grid_size=(int(args.clahe_grid[0]), int(args.clahe_grid[1])),
	)
	plot_class_distribution(dataset.labels, output_path=output_dirs["figures"] / "class_distribution.png")

	preprocessing_time_seconds = 0.0
	feature_extraction_time_seconds = 0.0
	svm_training_time_seconds = 0.0
	rf_training_time_seconds = 0.0
	svm_inference_time_per_image_seconds = 0.0
	rf_inference_time_per_image_seconds = 0.0

	# images are preprocessed during loading now; reuse directly
	processed_images = list(dataset.images)

	feature_start = perf_counter()
	features = prepare_feature_matrix(processed_images, hog_parameters=hog_parameters, use_glcm=args.use_glcm, use_orb=args.use_orb)
	feature_extraction_time_seconds = perf_counter() - feature_start
	x_train, x_test, y_train, y_test = split_dataset(features, dataset.labels, test_size=args.test_size)

	model = build_svm_classifier(svm_config)
	try:
		svm_train_start = perf_counter()
		train_classifier(model, x_train, y_train)
		svm_training_time_seconds = perf_counter() - svm_train_start
	except Exception as exc:
		print(f"[ERROR] SVM training failed: {exc}")
		svm_training_time_seconds = 0.0

	# Evaluate SVM baseline
	try:
		svm_inference_start = perf_counter()
		y_pred = predict_features(model, x_test)
		svm_inference_total = perf_counter() - svm_inference_start
		svm_inference_time_per_image_seconds = svm_inference_total / max(len(x_test), 1)
		svm_metrics = compute_metrics(y_test, y_pred)
	except Exception as exc:
		print(f"[ERROR] SVM evaluation failed: {exc}")
		y_pred = np.array([])
		svm_metrics = {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}

	# Train and evaluate Random Forest as an alternative classifier
	try:
		rf_model = build_random_forest_classifier()
		rf_train_start = perf_counter()
		train_classifier(rf_model, x_train, y_train)
		rf_training_time_seconds = perf_counter() - rf_train_start
		rf_inference_start = perf_counter()
		rf_y_pred = predict_features(rf_model, x_test)
		rf_inference_total = perf_counter() - rf_inference_start
		rf_inference_time_per_image_seconds = rf_inference_total / max(len(x_test), 1)
		rf_metrics = compute_metrics(y_test, rf_y_pred)
	except Exception as exc:
		print(f"[WARN] Random Forest training/evaluation failed: {exc}")
		rf_metrics = {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}

	# Save confusion matrices for both methods
	# Ensure consistent class order for binary labels: no_tumour, tumour
	present = set(dataset.labels)
	class_names = [c for c in ("no_tumour", "tumour") if c in present]
	figure = plot_confusion_matrix(y_test, y_pred, class_names, output_path=output_dirs["figures"] / "confusion_matrix_svm.png")
	plt.close(figure)
	figure = plot_confusion_matrix(y_test, rf_y_pred, class_names, output_path=output_dirs["figures"] / "confusion_matrix_rf.png")
	plt.close(figure)

	# Save classification reports and scalar metrics
	(report_dir := output_dirs["reports"]).mkdir(parents=True, exist_ok=True)
	(report_dir / "classification_report_svm.txt").write_text(generate_classification_report_text(y_test, y_pred), encoding="utf-8")
	(report_dir / "classification_report_rf.txt").write_text(generate_classification_report_text(y_test, rf_y_pred), encoding="utf-8")
	(report_dir / "metrics_svm.txt").write_text(format_metric_summary(svm_metrics), encoding="utf-8")
	(report_dir / "metrics_rf.txt").write_text(format_metric_summary(rf_metrics), encoding="utf-8")
	(report_dir / "timing.txt").write_text(
		format_timing_summary(
			{
				"preprocessing_time_seconds": preprocessing_time_seconds,
				"feature_extraction_time_seconds": feature_extraction_time_seconds,
				"svm_training_time_seconds": svm_training_time_seconds,
				"rf_training_time_seconds": rf_training_time_seconds,
				"svm_inference_time_per_image_seconds": svm_inference_time_per_image_seconds,
				"rf_inference_time_per_image_seconds": rf_inference_time_per_image_seconds,
			}
		),
		encoding="utf-8",
	)

	# Save a multi-method comparison report
	method_metrics = {"HOG + SVM": svm_metrics, "Random Forest": rf_metrics}
	save_multi_method_comparison_report_with_timing(
		method_metrics,
		output_dirs["comparisons"] / "method_comparison.md",
		{
			"preprocessing_time_seconds": preprocessing_time_seconds,
			"feature_extraction_time_seconds": feature_extraction_time_seconds,
			"svm_training_time_seconds": svm_training_time_seconds,
			"rf_training_time_seconds": rf_training_time_seconds,
			"svm_inference_time_per_image_seconds": svm_inference_time_per_image_seconds,
			"rf_inference_time_per_image_seconds": rf_inference_time_per_image_seconds,
		},
	)
	save_efficiency_report(
		{
			"preprocessing_time_seconds": preprocessing_time_seconds,
			"feature_extraction_time_seconds": feature_extraction_time_seconds,
			"svm_training_time_seconds": svm_training_time_seconds,
			"rf_training_time_seconds": rf_training_time_seconds,
			"svm_inference_time_per_image_seconds": svm_inference_time_per_image_seconds,
			"rf_inference_time_per_image_seconds": rf_inference_time_per_image_seconds,
		},
		output_dirs["comparisons"] / "efficiency_metrics.md",
	)

	# Save models
	save_model(model, args.model_path)
	save_model(rf_model, args.rf_model_path)

	# Prepare sample paths for annotated predictions
	if args.sample_dir is not None:
		sample_root = Path(args.sample_dir)
		sample_paths = [
			path
			for path in sample_root.rglob("*")
			if path.is_file() and path.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
		]
		sample_labels = []
		for path in sample_paths:
			try:
				sample_labels.append(infer_label_from_path(path, sample_root))
			except Exception:
				sample_labels.append(path.parent.name)
	else:
		sample_paths = list(dataset.file_paths)
		sample_labels = list(dataset.labels)

	sample_paths = select_diverse_sample_paths(sample_paths, sample_labels, args.num_sample_predictions)

	# Save annotated predictions for both models in separate subfolders
	pred_svm_dir = output_dirs["predictions"] / "svm"
	pred_rf_dir = output_dirs["predictions"] / "rf"
	pred_svm_dir.mkdir(parents=True, exist_ok=True)
	pred_rf_dir.mkdir(parents=True, exist_ok=True)

	annotated_svm = save_annotated_predictions(
		model,
		sample_paths,
		pred_svm_dir,
		output_dirs["figures"],
		hog_parameters=hog_parameters,
		segmentation_method=args.segmentation_method,
		use_glcm=args.use_glcm,
		use_orb=args.use_orb,
		use_clahe=True,
		clahe_clip=float(args.clahe_clip),
		clahe_grid=(int(args.clahe_grid[0]), int(args.clahe_grid[1])),
	)

	annotated_rf = save_annotated_predictions(
		rf_model,
		sample_paths,
		pred_rf_dir,
		output_dirs["figures"],
		hog_parameters=hog_parameters,
		segmentation_method=args.segmentation_method,
		use_glcm=args.use_glcm,
		use_orb=args.use_orb,
		use_clahe=True,
		clahe_clip=float(args.clahe_clip),
		clahe_grid=(int(args.clahe_grid[0]), int(args.clahe_grid[1])),
	)

	return {
		"svm_metrics": svm_metrics,
		"rf_metrics": rf_metrics,
		"timing": {
			"preprocessing_time_seconds": preprocessing_time_seconds,
			"feature_extraction_time_seconds": feature_extraction_time_seconds,
			"svm_training_time_seconds": svm_training_time_seconds,
			"rf_training_time_seconds": rf_training_time_seconds,
			"svm_inference_time_per_image_seconds": svm_inference_time_per_image_seconds,
			"rf_inference_time_per_image_seconds": rf_inference_time_per_image_seconds,
		},
		"svm_model_path": args.model_path,
		"rf_model_path": args.rf_model_path,
		"annotated_svm": annotated_svm,
		"annotated_rf": annotated_rf,
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

	print("Timing summary:")
	print(f"  Preprocessing time: {results['timing']['preprocessing_time_seconds']:.4f} s")
	print(f"  Feature extraction time: {results['timing']['feature_extraction_time_seconds']:.4f} s")
	print(f"  SVM training time: {results['timing']['svm_training_time_seconds']:.4f} s")
	print(f"  RF training time: {results['timing']['rf_training_time_seconds']:.4f} s")
	print(f"  SVM inference time / image: {results['timing']['svm_inference_time_per_image_seconds']:.6f} s")
	print(f"  RF inference time / image: {results['timing']['rf_inference_time_per_image_seconds']:.6f} s")

	print("Annotated predictions (SVM):")
	for path in results['annotated_svm']:
		print(f"  - {path}")
	print("Annotated predictions (RF):")
	for path in results['annotated_rf']:
		print(f"  - {path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
