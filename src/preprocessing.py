"""Image loading and preprocessing steps."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config import ProjectConfig


def load_original_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def preprocess_image(image: np.ndarray, config: ProjectConfig) -> np.ndarray:
    grayscale_image = convert_to_grayscale(image)
    blurred_image = cv2.GaussianBlur(grayscale_image, config.gaussian_kernel, 0)
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid_size,
    )
    enhanced_image = clahe.apply(blurred_image)
    resized_image = cv2.resize(enhanced_image, config.image_size, interpolation=cv2.INTER_AREA)
    return resized_image
