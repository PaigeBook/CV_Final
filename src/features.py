"""Feature extraction helpers."""

from __future__ import annotations

import numpy as np
from skimage.feature import hog


def extract_hog_features(
    image: np.ndarray,
    orientations: int,
    pixels_per_cell: tuple[int, int],
    cells_per_block: tuple[int, int],
) -> np.ndarray:
    feature_vector = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )
    return np.asarray(feature_vector, dtype=np.float32)
