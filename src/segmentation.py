"""Basic segmentation utilities for highlighting suspicious regions."""

from __future__ import annotations

import cv2
import numpy as np


def segment_suspicious_region(
    enhanced_image: np.ndarray,
    morph_kernel_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a binary mask and contour overlay for the strongest bright region."""

    if enhanced_image.ndim != 2:
        raise ValueError("Segmentation expects a grayscale enhanced image.")

    _, threshold_mask = cv2.threshold(
        enhanced_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = np.ones((morph_kernel_size, morph_kernel_size), dtype=np.uint8)
    cleaned_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_overlay = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
    suspicious_mask = np.zeros_like(enhanced_image, dtype=np.uint8)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 0:
            cv2.drawContours(contour_overlay, [largest_contour], -1, (0, 0, 255), 2)
            cv2.drawContours(suspicious_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return cleaned_mask, contour_overlay, suspicious_mask
