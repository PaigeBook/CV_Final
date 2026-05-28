"""Model training and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class ModelSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    random_state: int,
) -> ModelSplit:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return ModelSplit(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def build_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", class_weight="balanced", random_state=42)),
        ]
    )


def evaluate_classifier(model: Pipeline, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, object]:
    predictions = model.predict(x_test)
    return {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, predictions, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "predictions": predictions,
    }
