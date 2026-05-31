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
        PreprocessingSettings,
        SVMConfig,
        build_baseline_classifier,
        build_random_forest_classifier,
        build_svm_classifier,
        infer_label_from_path,
        load_dataset,
        load_model,
        predict_features,
        predict_features_with_confidence,
        predict_image,
        prepare_feature_matrix,
        prepare_raw_pixel_matrix,
        save_model,
        split_dataset,
        train_classifier,
    )
    from .evaluation import (
        compute_metrics,
        format_metric_summary,
        format_timing_summary,
        plot_confusion_matrix,
    )
    from .feature_extraction import HOGParameters
    from .preprocessing import load_original_image, preprocess_image
    from .segmentation import segment_defects
    from .utils import prepare_dataset_directory, select_diverse_sample_paths
    from .viz import save_hog_visualisations
    from .visualization import save_case_artifacts
except ImportError:  # pragma: no cover - fallback for direct script execution
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.classifier import (  # type: ignore
        PreprocessingSettings,
        SVMConfig,
        build_baseline_classifier,
        build_random_forest_classifier,
        build_svm_classifier,
        infer_label_from_path,
        load_dataset,
        load_model,
        predict_features,
        predict_features_with_confidence,
        predict_image,
        prepare_feature_matrix,
        prepare_raw_pixel_matrix,
        save_model,
        split_dataset,
        train_classifier,
    )
    from src.evaluation import (  # type: ignore
        compute_metrics,
        format_metric_summary,
        format_timing_summary,
        plot_confusion_matrix,
    )
    from src.feature_extraction import HOGParameters  # type: ignore
    from src.preprocessing import load_original_image, preprocess_image  # type: ignore
    from src.segmentation import segment_defects  # type: ignore
    from src.utils import prepare_dataset_directory, select_diverse_sample_paths  # type: ignore
    from src.viz import save_hog_visualisations  # type: ignore
    from src.visualization import save_case_artifacts  # type: ignore


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
	parser.add_argument("--morph-kernel-size", type=int, default=5, help="Morphological kernel size used during ROI segmentation")
	parser.add_argument("--skip-baseline", action="store_true", help="Skip training the raw-pixel baseline model")
	parser.add_argument("--force-train", action="store_true", help="Retrain models even if saved versions already exist")
	parser.add_argument("--generate-assets", action="store_true", help="Generate false-negative, gallery, and HOG review assets")
	parser.add_argument("--preserve-outputs", action="store_true", help="Skip automatic clearing of generated output folders at startup")
	parser.add_argument("--reset-reports", action="store_true", help="Delete output reports before running to avoid accumulated files")
	parser.add_argument("--max-false-negatives", type=int, default=10, help="Maximum false-negative examples to export")
	parser.add_argument("--num-sample-predictions", type=int, default=3, help="Number of annotated sample images to save")
	return parser


def ensure_output_directories(output_dir: Path) -> dict[str, Path]:
	"""Create output folders."""

	evaluation_dir = output_dir / "evaluation"
	predictions_dir = output_dir / "predictions"
	predictions_correct_dir = predictions_dir / "correct"
	predictions_incorrect_dir = predictions_dir / "incorrect"
	features_dir = output_dir / "features"
	hog_examples_dir = features_dir / "hog_examples"
	runtime_dir = output_dir / "runtime"

	for d in (output_dir, evaluation_dir, predictions_dir, predictions_correct_dir, predictions_incorrect_dir, features_dir, hog_examples_dir, runtime_dir):
		d.mkdir(parents=True, exist_ok=True)

	paths: dict[str, Path] = {
		"root": output_dir,
		"evaluation": evaluation_dir,
		"predictions": predictions_dir,
		"predictions_correct": predictions_correct_dir,
		"predictions_incorrect": predictions_incorrect_dir,
		"features": features_dir,
		"hog_examples": hog_examples_dir,
		"runtime": runtime_dir,
	}

	# Backwards-compatible aliases used by older code paths.
	paths["reports"] = evaluation_dir
	paths["comparisons"] = evaluation_dir
	paths["gallery"] = predictions_dir
	paths["misclassified"] = predictions_incorrect_dir
	paths["figures"] = predictions_dir
	paths["cases"] = predictions_dir
	paths["false_negatives"] = predictions_incorrect_dir

	return paths


def clear_generated_output_contents(output_dirs: dict[str, Path]) -> None:
	"""Remove generated files from prior runs while preserving the folder tree."""

	print("[INFO] Clearing previous generated outputs...")
	legacy_root_dirs = ["cases", "gallery", "misclassified", "reports", "logs", "models"]
	for legacy_name in legacy_root_dirs:
		legacy_path = output_dirs["root"] / legacy_name
		if legacy_path.exists():
			shutil.rmtree(legacy_path)

	paths_to_clear = [
		output_dirs["evaluation"],
		output_dirs["predictions"],
		output_dirs["predictions_correct"],
		output_dirs["predictions_incorrect"],
		output_dirs["features"],
		output_dirs["hog_examples"],
		output_dirs["runtime"],
	]

	for directory in paths_to_clear:
		directory.mkdir(parents=True, exist_ok=True)
		for child in list(directory.iterdir()):
			if child.is_dir():
				shutil.rmtree(child)
			else:
				child.unlink()


# prepare_dataset_directory moved to src.utils


# overlay_label moved to src.viz


def build_hog_parameters(args: argparse.Namespace) -> HOGParameters:
	"""Build HOG parameters from CLI args."""

	return HOGParameters(
		resize_to=(int(args.hog_size[0]), int(args.hog_size[1])),
	)


def build_svm_config(args: argparse.Namespace, hog_parameters: HOGParameters) -> SVMConfig:
	"""Build SVM config from CLI args."""

	return SVMConfig(hog_parameters=hog_parameters)


def build_preprocessing_settings(args: argparse.Namespace) -> PreprocessingSettings:
	"""Build the preprocessing settings used throughout the pipeline."""

	return PreprocessingSettings(
		image_size=(int(args.hog_size[0]), int(args.hog_size[1])),
		gaussian_kernel=(5, 5),
		clahe_clip_limit=2.0,
		clahe_tile_grid_size=(8, 8),
	)


# select_diverse_sample_paths moved to src.utils


# save_visual_summary moved to src.viz


# save_annotated_predictions moved to src.viz


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
	"""Run training, evaluation, and output generation."""

	output_dirs = ensure_output_directories(args.output_dir)
	if not getattr(args, "preserve_outputs", False):
		clear_generated_output_contents(output_dirs)
		if getattr(args, "reset_reports", False):
			reports_to_clear = output_dirs["reports"]
			for child in list(reports_to_clear.iterdir()):
				if child.is_dir():
					shutil.rmtree(child)
				else:
					child.unlink()
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
	# Build concatenated HOG features: full enhanced image + ROI-masked image
	hog_features = prepare_feature_matrix(hog_ready_images, hog_parameters=hog_parameters, full_images=preprocessed_images)
	feature_extraction_time_seconds = perf_counter() - feature_start

	raw_feature_start = perf_counter()
	raw_pixel_features = prepare_raw_pixel_matrix(raw_pixel_images, resize_to=(128, 128))
	baseline_feature_extraction_time_seconds = perf_counter() - raw_feature_start

	split = split_dataset(hog_features, dataset.labels, test_size=args.test_size, random_state=42)
	train_indices = split.train_indices
	test_indices = split.test_indices
	raw_x_test = raw_pixel_features[test_indices]
	image_paths_test = [dataset.file_paths[int(index)] for index in test_indices]
	original_test_images = [dataset.images[int(index)] for index in test_indices]
	test_masks = [suspicious_masks[int(index)] for index in test_indices]
	preprocessed_test_images = [preprocessed_images[int(index)] for index in test_indices]

	svm_model_path = Path(args.model_path)
	rf_model_path = Path(args.rf_model_path)
	baseline_model_path = Path(args.baseline_model_path)
	svm_model = None
	rf_model = None
	baseline_model = None
	svm_trained = False
	rf_trained = False
	baseline_trained = False

	if not args.force_train and svm_model_path.exists():
		try:
			svm_model = load_model(svm_model_path)
			print(f"Loaded existing SVM model from: {svm_model_path}")
		except Exception as exc:
			print(f"[WARN] Failed to load SVM model, retraining instead: {exc}")
	if svm_model is None:
		svm_model = build_svm_classifier(svm_config)
		try:
			svm_train_start = perf_counter()
			train_classifier(svm_model, split.x_train, split.y_train)
			svm_training_time_seconds = perf_counter() - svm_train_start
			svm_trained = True
		except Exception as exc:
			print(f"[ERROR] SVM training failed: {exc}")
			svm_training_time_seconds = 0.0

	# Evaluate the HOG + SVM model on the shared test split.
	try:
		svm_inference_start = perf_counter()
		svm_predictions, svm_confidences = predict_features_with_confidence(svm_model, split.x_test)
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
	rf_y_pred = np.array([])
	if not args.force_train and rf_model_path.exists():
		try:
			rf_model = load_model(rf_model_path)
			print(f"Loaded existing Random Forest model from: {rf_model_path}")
		except Exception as exc:
			print(f"[WARN] Failed to load Random Forest model, retraining instead: {exc}")
	if rf_model is None:
		try:
			rf_model = build_random_forest_classifier()
			rf_train_start = perf_counter()
			train_classifier(rf_model, split.x_train, split.y_train)
			rf_training_time_seconds = perf_counter() - rf_train_start
			rf_trained = True
		except Exception as exc:
			print(f"[WARN] Random Forest training/evaluation failed: {exc}")
	if rf_model is not None:
		try:
			rf_inference_start = perf_counter()
			rf_y_pred = predict_features(rf_model, split.x_test)
			rf_inference_total = perf_counter() - rf_inference_start
			rf_inference_time_per_image_seconds = rf_inference_total / max(len(split.x_test), 1)
			rf_metrics = compute_metrics(split.y_test, rf_y_pred)
		except Exception as exc:
			print(f"[WARN] Random Forest evaluation failed: {exc}")

	baseline_metrics: dict[str, object] | None = None
	baseline_y_pred: np.ndarray | None = None
	if not args.skip_baseline:
		if not args.force_train and baseline_model_path.exists():
			try:
				baseline_model = load_model(baseline_model_path)
				print(f"Loaded existing baseline model from: {baseline_model_path}")
			except Exception as exc:
				print(f"[WARN] Failed to load baseline model, retraining instead: {exc}")
		if baseline_model is None:
			try:
				baseline_model = build_baseline_classifier()
				baseline_train_start = perf_counter()
				train_classifier(baseline_model, raw_pixel_features[train_indices], split.y_train)
				baseline_training_time_seconds = perf_counter() - baseline_train_start
				baseline_trained = True
			except Exception as exc:
				print(f"[WARN] Baseline training/evaluation failed: {exc}")
		if baseline_model is not None:
			try:
				baseline_inference_start = perf_counter()
				baseline_y_pred = predict_features(baseline_model, raw_x_test)
				baseline_inference_total = perf_counter() - baseline_inference_start
				baseline_inference_time_per_image_seconds = baseline_inference_total / max(len(split.x_test), 1)
				baseline_metrics = compute_metrics(split.y_test, baseline_y_pred)
			except Exception as exc:
				print(f"[WARN] Baseline evaluation failed: {exc}")
				baseline_metrics = {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}

	# Save evaluation outputs in the consolidated evaluation folder.
	present = set(dataset.labels.tolist())
	class_names = [c for c in ("no_tumour", "tumour") if c in present]
	figure = plot_confusion_matrix(split.y_test, svm_predictions, class_names, output_path=output_dirs["evaluation"] / "confusion_matrix_svm.png")
	plt.close(figure)
	if rf_y_pred.size:
		figure = plot_confusion_matrix(split.y_test, rf_y_pred, class_names, output_path=output_dirs["evaluation"] / "confusion_matrix_rf.png")
		plt.close(figure)
	evaluation_text_lines = [
		"# Evaluation Summary\n\n",
		"## HOG + SVM\n\n",
		format_metric_summary(svm_metrics),
		f"Confusion matrix: {svm_metrics.get('confusion_matrix', [])}\n\n",
		"## HOG + Random Forest\n\n",
		format_metric_summary(rf_metrics),
		f"Confusion matrix: {rf_metrics.get('confusion_matrix', [])}\n",
	]
	(output_dirs["evaluation"] / "metrics.txt").write_text("".join(evaluation_text_lines), encoding="utf-8")

	# Persist models only after fresh training.
	if svm_trained:
		save_model(svm_model, svm_model_path)
	if rf_trained:
		save_model(rf_model, rf_model_path)
	if baseline_model is not None and baseline_trained:
		save_model(baseline_model, baseline_model_path)

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
			feature_vector = prepare_feature_matrix([masked_image], hog_parameters=hog_parameters, full_images=[enhanced_image])
			timing_features += perf_counter() - stage_start

			stage_start = perf_counter()
			predict_features_with_confidence(svm_model, feature_vector)
			timing_classify += perf_counter() - stage_start

		preprocessing_time_seconds = timing_preprocess
		segmentation_time_seconds = timing_segment
		feature_extraction_time_seconds = timing_features
		classification_time_seconds = timing_classify
		total_inference_time_per_image_seconds = (
			preprocessing_time_seconds + segmentation_time_seconds + feature_extraction_time_seconds + classification_time_seconds
		) / max(len(test_indices), 1)

	prediction_paths: list[Path] = []
	hog_visual_paths: list[Path] = []
	if args.generate_assets:
		prediction_sample_paths = select_diverse_sample_paths(image_paths_test, split.y_test, max(1, int(args.num_sample_predictions)))
		for index, image_path in enumerate(prediction_sample_paths, start=1):
			original_image = load_original_image(image_path)
			enhanced_image = preprocess_image(original_image, preprocessing_settings)
			segmentation_mask, contour_overlay, suspicious_mask = segment_defects(enhanced_image, args.morph_kernel_size)
			masked_image = enhanced_image.copy()
			masked_image[suspicious_mask == 0] = 0
			predicted_label, confidence = predict_image(
				svm_model,
				masked_image,
				hog_parameters=hog_parameters,
				full_image=enhanced_image,
			)
			ground_truth_label = infer_label_from_path(image_path, dataset_dir)
			prediction_folder = output_dirs["predictions_correct"] if predicted_label == ground_truth_label else output_dirs["predictions_incorrect"]
			prediction_path = prediction_folder / f"prediction_{index:02d}_{image_path.stem}_pred-{predicted_label}_conf-{confidence:.3f}.png"
			save_case_artifacts(
				original_image=original_image,
				enhanced_image=enhanced_image,
				segmentation_mask=segmentation_mask,
				contour_overlay=contour_overlay,
				predicted_label=predicted_label,
				ground_truth_label=ground_truth_label,
				confidence=confidence,
					output_path=prediction_path,
			)
			prediction_paths.append(prediction_path)
		feature_sample_paths: list[Path] = []
		for label in ("tumour", "no_tumour"):
			label_paths = [path for path, sample_label in zip(image_paths_test, split.y_test) if sample_label == label]
			label_samples = select_diverse_sample_paths(label_paths, [label] * len(label_paths), min(3, len(label_paths)))
			feature_sample_paths.extend(label_samples)
		if not feature_sample_paths:
			feature_sample_paths = select_diverse_sample_paths(image_paths_test, split.y_test, min(6, max(1, int(args.num_sample_predictions) * 2)))
		feature_visual_paths = save_hog_visualisations(
			feature_sample_paths,
			output_dirs["hog_examples"],
			max_examples=6,
			hog_parameters=hog_parameters,
		)
		hog_visual_paths = feature_visual_paths
	else:
		print("Asset generation skipped; use --generate-assets to export prediction composites and HOG visuals.")

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
	(output_dirs["runtime"] / "runtime_summary.txt").write_text(format_timing_summary(timing_summary), encoding="utf-8")

	return {
		"svm_metrics": svm_metrics,
		"rf_metrics": rf_metrics,
		"baseline_metrics": baseline_metrics,
		"timing": timing_summary,
		"svm_model_path": args.model_path,
		"rf_model_path": args.rf_model_path,
		"baseline_model_path": args.baseline_model_path,
		"prediction_paths": prediction_paths,
		"feature_visual_paths": hog_visual_paths,
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
	print(f"Prediction composites saved: {len(results['prediction_paths'])}")
	print(f"HOG feature visualisations saved: {len(results['feature_visual_paths'])}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
