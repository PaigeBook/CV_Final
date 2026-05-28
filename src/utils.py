"""Small utility helpers used by the CLI launcher."""

from __future__ import annotations

from pathlib import Path
import zipfile
import numpy as np
import shutil
from typing import Iterable, List


def prepare_dataset_directory(dataset_arg: Path | None, project_root: Path) -> Path:
    """Resolve and prepare the dataset directory for the pipeline.

    If a dataset path is provided, it is returned. Otherwise, the function will
    look for a zip archive in the project root and extract it to
    `data/raw/Brain_Tumor_Dataset` if necessary.
    """

    if dataset_arg is not None and dataset_arg.exists():
        return dataset_arg

    default_zip = project_root / "Brain_Tumor_Dataset.zip"
    default_dir = project_root / "data" / "raw" / "Brain_Tumor_Dataset"
    if default_dir.exists():
        return default_dir

    if default_zip.exists():
        default_dir.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(default_zip, "r") as archive:
            archive.extractall(default_dir.parent)
        return default_dir

    # Fall back to a commonly used relative path used by older launchers
    fallback = project_root / "data" / "raw" / "mri"
    if fallback.exists():
        return fallback

    raise FileNotFoundError("Dataset not found. Place the dataset under data/raw/ or add the zip to the project root.")


def select_diverse_sample_paths(paths: Iterable[Path], labels: Iterable[str], num: int) -> List[Path]:
    """Select up to `num` diverse sample paths from the provided lists.

    The function attempts to balance samples across classes when possible so the
    visual gallery is informative.
    """

    paths = list(paths)
    labels = list(labels)
    if len(paths) <= num:
        return paths

    by_label: dict[str, list[Path]] = {}
    for p, l in zip(paths, labels):
        by_label.setdefault(l, []).append(p)

    selected: list[Path] = []
    per_label = max(1, num // max(1, len(by_label)))
    rng = np.random.default_rng(42)

    for label, group in by_label.items():
        choices = list(group)
        if len(choices) <= per_label:
            selected.extend(choices)
        else:
            indices = rng.choice(len(choices), size=per_label, replace=False)
            selected.extend([choices[i] for i in indices])

    # Fill remaining slots randomly
    remaining = [p for p in paths if p not in selected]
    if len(selected) < num and remaining:
        need = num - len(selected)
        indices = rng.choice(len(remaining), size=min(need, len(remaining)), replace=False)
        selected.extend([remaining[i] for i in indices])

    return selected[:num]
