"""Segmentation utilities for highlighting suspicious regions.

The goal here is not medical-grade lesion segmentation. The pipeline only needs
stable region-of-interest estimation so the classical feature extractor can
focus on the most informative tissue area.
"""

from __future__ import annotations

import cv2
import numpy as np


def overlay_mask(image: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int] = (0, 0, 255), alpha: float = 0.4) -> np.ndarray:
    """Overlay a binary mask on top of an image."""

    base_image = image.copy()
    if base_image.ndim == 2:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    overlay = np.zeros_like(base_image, dtype=np.uint8)
    overlay[mask > 0] = color
    return cv2.addWeighted(base_image, 1.0, overlay, float(alpha), 0.0)


def _remove_small_components(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    """Remove tiny connected components from a binary mask."""

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def segment_suspicious_region(
    enhanced_image: np.ndarray,
    morph_kernel_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the region of interest using a robust classical pipeline.

    Steps:
    1. Blur to suppress pixel noise.
    2. Apply Otsu thresholding to derive a coarse foreground mask.
    3. Clean the mask with opening and closing.
    4. Remove tiny connected components.
    5. Keep only the largest remaining contour as the region of interest.
    """

    if enhanced_image.ndim != 2:
        raise ValueError("Segmentation expects a grayscale enhanced image.")

    blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
    _, threshold_mask = cv2.threshold(
        blurred_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = np.ones((morph_kernel_size, morph_kernel_size), dtype=np.uint8)
    cleaned_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = _remove_small_components(cleaned_mask, min_area=max(20, int(enhanced_image.size * 0.002)))

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_overlay = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
    suspicious_mask = np.zeros_like(enhanced_image, dtype=np.uint8)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 0:
            cv2.drawContours(contour_overlay, [largest_contour], -1, (0, 0, 255), 2)
            cv2.drawContours(suspicious_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return cleaned_mask, contour_overlay, suspicious_mask


def segment_defects(enhanced_image: np.ndarray, morph_kernel_size: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible wrapper used by older parts of the project."""

    return segment_suspicious_region(enhanced_image, morph_kernel_size)


def annotate_defects(
    image: np.ndarray,
    segmentation_mask: np.ndarray,
    *,
    title: str | None = None,
    confidence: float | None = None,
) -> np.ndarray:
    """Create a human-readable overlay for a segmented MRI image."""

    annotated = overlay_mask(image, segmentation_mask)
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

    banner = annotated.copy()
    text = title or "ROI"
    if confidence is not None:
        text = f"{text} | confidence: {confidence:.3f}"
    cv2.rectangle(banner, (0, 0), (banner.shape[1], 36), (0, 0, 0), thickness=-1)
    cv2.putText(banner, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return banner
