"""Classifier utilities for the classical MRI pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data_loader import collect_image_records, infer_class_name, infer_label
from .feature_extraction import HOGParameters
from .feature_extraction import (
    extract_glcm_features,
    extract_hog_features_for_prediction,
    extract_orb_features,
    HOGParameters,
)
from .preprocessing import convert_to_grayscale, load_original_image, preprocess_image


IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True, slots=True)
class PreprocessingSettings:
    """Lightweight preprocessing settings used when loading the dataset."""

    image_size: tuple[int, int] = (256, 256)
    gaussian_kernel: tuple[int, int] = (5, 5)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """Loaded dataset and file metadata."""

    images: list[np.ndarray]
    labels: np.ndarray
    file_paths: list[Path]
    class_names: list[str]


@dataclass(frozen=True, slots=True)
class ModelSplit:
    """Train/test split with indices preserved for fair comparisons."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class SVMConfig:
    """Configuration for the HOG + SVM classifier."""

    kernel: str = "rbf"
    c: float = 5.0
    random_state: int = 42
    hog_parameters: HOGParameters = HOGParameters()


def infer_label_from_path(image_path: Path, dataset_root: Path) -> str:
    """Infer a canonical label name from a file path."""

    class_name = infer_class_name(image_path, dataset_root)
    label_value = infer_label(class_name)
    return "tumour" if label_value == 1 else "no_tumour"


def load_dataset(
    dataset_dir: Path,
    *,
    preprocess: bool = True,
    preprocessing_settings: PreprocessingSettings | None = None,
) -> DatasetBundle:
    """Load MRI images and labels from the dataset folder.

    When ``preprocess`` is true, images are converted to grayscale, denoised,
    contrast-enhanced, and resized so the feature extraction stage can work on a
    stable representation.
    """

    records = collect_image_records(dataset_dir)
    settings = preprocessing_settings or PreprocessingSettings()
    loaded_images: list[np.ndarray] = []
    labels: list[str] = []
    class_names: list[str] = []

    for record in records:
        original_image = load_original_image(record.image_path)
        if preprocess:
            image = preprocess_image(original_image, settings)
        else:
            image = original_image
        loaded_images.append(image)
        labels.append("tumour" if record.label == 1 else "no_tumour")
        class_names.append(record.class_name)

    return DatasetBundle(
        images=loaded_images,
        labels=np.asarray(labels, dtype=object),
        file_paths=[record.image_path for record in records],
        class_names=class_names,
    )


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    random_state: int,
) -> ModelSplit:
    """Create a stratified split and preserve the row indices.

    The indices are reused for the HOG model, random forest, and baseline raw
    pixel model so that all methods are compared on exactly the same samples.
    """

    feature_indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        feature_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return ModelSplit(
        x_train=features[train_indices],
        x_test=features[test_indices],
        y_train=labels[train_indices],
        y_test=labels[test_indices],
        train_indices=train_indices,
        test_indices=test_indices,
    )


def build_svm_classifier(config: SVMConfig) -> Pipeline:
    """Build the HOG + SVM model."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel=config.kernel,
                    C=config.c,
                    class_weight="balanced",
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def build_random_forest_classifier(random_state: int = 42) -> RandomForestClassifier:
    """Build the HOG + Random Forest model."""

    return RandomForestClassifier(
        n_estimators=150,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )


def build_baseline_classifier(random_state: int = 42) -> Pipeline:
    """Build a baseline classifier on flattened grayscale pixels."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_model_registry(config: SVMConfig, *, random_state: int = 42) -> dict[str, Any]:
    """Return the project model suite in a single registry."""

    return {
        "hog_svm": build_svm_classifier(config),
        "hog_random_forest": build_random_forest_classifier(random_state=random_state),
        "raw_logistic_regression": build_baseline_classifier(random_state=random_state),
    }


def train_classifier(model: Any, x_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Fit a scikit-learn model and return it for chaining."""

    model.fit(x_train, y_train)
    return model


def predict_features(model: Any, features: np.ndarray) -> np.ndarray:
    """Predict labels for a feature matrix."""

    return np.asarray(model.predict(features))


def predict_features_with_confidence(model: Any, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels and confidence scores for a feature matrix.

    The SVM pipeline exposes ``decision_function``; other estimators fall back to
    ``predict_proba`` when available. The returned score is intended for medical
    review and should be interpreted as a relative confidence measure, not a
    calibrated probability unless the estimator supports it.
    """

    predictions = np.asarray(model.predict(features))

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=np.float32)
        if scores.ndim > 1:
            scores = scores.max(axis=1)
        return predictions, scores

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=np.float32)
        scores = probabilities.max(axis=1)
        return predictions, scores

    return predictions, np.ones(len(predictions), dtype=np.float32)


def predict_image(
    model: Any,
    image: np.ndarray | str | Path,
    *,
    hog_parameters: HOGParameters | None = None,
    use_glcm: bool = False,
    use_orb: bool = False,
    raw_pixel_size: tuple[int, int] | None = None,
) -> tuple[str, float]:
    """Predict a single image and return the label with confidence."""

    if isinstance(image, (str, Path)):
        image_array = load_original_image(Path(image))
    else:
        image_array = image

    if raw_pixel_size is not None:
        grayscale = convert_to_grayscale(image_array)
        resized = cv2.resize(grayscale, raw_pixel_size, interpolation=cv2.INTER_AREA)
        features = resized.reshape(1, -1).astype(np.float32) / 255.0
    else:
        features = prepare_feature_matrix([image_array], hog_parameters=hog_parameters, use_glcm=use_glcm, use_orb=use_orb)

    prediction, scores = predict_features_with_confidence(model, features)
    return str(prediction[0]), float(scores[0])


def prepare_feature_matrix(
    images: list[np.ndarray],
    *,
    hog_parameters: HOGParameters | None = None,
    use_glcm: bool = False,
    use_orb: bool = False,
) -> np.ndarray:
    """Build the HOG-based feature matrix used by the classical models."""

    params = hog_parameters or HOGParameters()
    rows: list[np.ndarray] = []
    for image in images:
        hog_features = extract_hog_features_for_prediction(image, params)
        parts: list[np.ndarray] = [hog_features]
        if use_orb:
            parts.append(extract_orb_features(image))
        if use_glcm:
            parts.append(extract_glcm_features(image))
        rows.append(np.concatenate(parts).astype(np.float32))
    return np.vstack(rows).astype(np.float32)


def prepare_raw_pixel_matrix(images: list[np.ndarray], *, resize_to: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Flatten grayscale images into a raw-pixel baseline matrix."""

    rows: list[np.ndarray] = []
    for image in images:
        grayscale = convert_to_grayscale(image)
        resized = cv2.resize(grayscale, resize_to, interpolation=cv2.INTER_AREA)
        rows.append((resized.reshape(-1).astype(np.float32)) / 255.0)
    return np.vstack(rows).astype(np.float32)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard scalar metrics for model comparison."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label="tumour", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label="tumour", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label="tumour", zero_division=0)),
    }


def save_model(model: Any, model_path: str | Path) -> Path:
    """Persist a trained model to disk."""

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
