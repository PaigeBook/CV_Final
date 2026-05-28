"""Dataset discovery and label inference utilities."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NEGATIVE_LABELS = {"negative", "no", "no_tumor", "no-tumor", "notumor", "normal", "healthy"}
POSITIVE_LABELS = {"positive", "yes", "tumor", "tumour", "abnormal"}


@dataclass(frozen=True)
class DatasetRecord:
    image_path: Path
    label: int
    class_name: str


def ensure_dataset_available(data_root: Path, zip_path: Path) -> Path:
    """Extract the dataset archive when the raw folder is not yet available."""

    if data_root.exists():
        return data_root

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found at {data_root} and zip archive is missing at {zip_path}."
        )

    data_root.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(data_root.parent)
    return data_root


def collect_image_records(data_root: Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for image_path in sorted(data_root.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        class_name = infer_class_name(image_path, data_root)
        label = infer_label(class_name)
        records.append(DatasetRecord(image_path=image_path, label=label, class_name=class_name))

    if not records:
        raise ValueError(f"No supported image files were found under {data_root}.")

    return records


def infer_class_name(image_path: Path, data_root: Path) -> str:
    relative_parts = image_path.relative_to(data_root).parts
    if len(relative_parts) < 2:
        return image_path.parent.name
    return relative_parts[0]


def infer_label(class_name: str) -> int:
    normalized = class_name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in NEGATIVE_LABELS:
        return 0
    if normalized in POSITIVE_LABELS:
        return 1
    if "no" in normalized or "neg" in normalized or "normal" in normalized:
        return 0
    if "tumor" in normalized or "tumour" in normalized or "pos" in normalized:
        return 1
    raise ValueError(f"Unable to infer class label from folder name: {class_name}")
