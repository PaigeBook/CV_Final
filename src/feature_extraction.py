"""Feature extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import hog
from skimage.feature.texture import graycomatrix, graycoprops


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
	if image.ndim == 2:
		return image
	if image.ndim != 3:
		raise ValueError("Expected a 2D grayscale or 3D colour image.")
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _resize_image(image: np.ndarray, resize_to: tuple[int, int] | None) -> np.ndarray:
	if resize_to is None:
		return image
	width, height = resize_to
	return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


@dataclass(frozen=True, slots=True)
class HOGParameters:
	"""Shared HOG settings."""

	resize_to: tuple[int, int] = (128, 128)
	orientations: int = 9
	pixels_per_cell: tuple[int, int] = (8, 8)
	cells_per_block: tuple[int, int] = (2, 2)
	transform_sqrt: bool = True
	block_norm: str = "L2-Hys"


def _compute_hog(image: np.ndarray, params: HOGParameters) -> np.ndarray:
	grayscale = _resize_image(_ensure_grayscale(image), params.resize_to)
	# Ensure resize dimensions are compatible with pixels_per_cell
	pw, ph = params.pixels_per_cell
	h, w = grayscale.shape[:2]
	adj_w = (w // pw) * pw
	adj_h = (h // ph) * ph
	if adj_w != w or adj_h != h:
		grayscale = cv2.resize(grayscale, (adj_w, adj_h), interpolation=cv2.INTER_AREA)
	features = hog(
		grayscale,
		orientations=params.orientations,
		pixels_per_cell=params.pixels_per_cell,
		cells_per_block=params.cells_per_block,
		transform_sqrt=params.transform_sqrt,
		block_norm=params.block_norm,
		feature_vector=True,
	)
	return np.asarray(features, dtype=np.float32)


def extract_hog_features(
	image: np.ndarray,
	*,
	resize_to: tuple[int, int] = (128, 128),
	orientations: int = 9,
	pixels_per_cell: tuple[int, int] = (8, 8),
	cells_per_block: tuple[int, int] = (2, 2),
) -> np.ndarray:
	params = HOGParameters(
		resize_to=resize_to,
		orientations=orientations,
		pixels_per_cell=pixels_per_cell,
		cells_per_block=cells_per_block,
	)
	return _compute_hog(image, params)


def extract_hog_features_for_prediction(image: np.ndarray, params: HOGParameters | None = None) -> np.ndarray:
	return _compute_hog(image, params or HOGParameters())


def extract_hog_features_for_training(images: list[np.ndarray], params: HOGParameters | None = None) -> np.ndarray:
	parameters = params or HOGParameters()
	feature_rows = [_compute_hog(image, parameters) for image in images]
	return np.vstack(feature_rows).astype(np.float32)


def build_hog_feature_matrix(images: list[np.ndarray], params: HOGParameters | None = None) -> np.ndarray:
	return extract_hog_features_for_training(images, params=params)


def extract_orb_features(
	image: np.ndarray,
	*,
	resize_to: tuple[int, int] = (256, 256),
	max_features: int = 500,
) -> np.ndarray:
	"""Extract a fixed-length ORB summary vector."""

	grayscale = _resize_image(_ensure_grayscale(image), resize_to)
	orb = cv2.ORB_create(nfeatures=max_features)
	keypoints, descriptors = orb.detectAndCompute(grayscale, None)

	if descriptors is None or len(descriptors) == 0:
		return np.zeros(65, dtype=np.float32)

	descriptors = descriptors.astype(np.float32)
	descriptor_mean = descriptors.mean(axis=0)
	descriptor_std = descriptors.std(axis=0)
	keypoint_count = np.array([len(keypoints)], dtype=np.float32)
	return np.concatenate([descriptor_mean, descriptor_std, keypoint_count]).astype(np.float32)


def extract_feature_vector(image: np.ndarray, *, use_orb: bool = False) -> np.ndarray:
	"""Build a feature vector for classification."""

	hog_features = extract_hog_features_for_prediction(image)
	if not use_orb:
		return hog_features
	orb_features = extract_orb_features(image)
	return np.concatenate([hog_features, orb_features]).astype(np.float32)


def build_feature_matrix(images: list[np.ndarray], *, use_orb: bool = False) -> np.ndarray:
	feature_rows = [extract_feature_vector(image, use_orb=use_orb) for image in images]
	return np.vstack(feature_rows).astype(np.float32)


def extract_glcm_features(
	image: np.ndarray,
	distances: tuple[int, ...] = (1, 2),
	angles: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
	levels: int = 256,
	properties: tuple[str, ...] = ("contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"),
) -> np.ndarray:
	"""Extract GLCM texture features (lightweight summary).

	Returns a 1D feature vector concatenating requested properties across distances and angles.
	"""

	grayscale = _ensure_grayscale(image)
	if grayscale.dtype != np.uint8:
		# scale to 0-255
		img = (255 * (grayscale.astype(np.float32) - grayscale.min()) / max(1e-8, grayscale.max() - grayscale.min())).astype(np.uint8)
	else:
		img = grayscale

	matrix = graycomatrix(img, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
	feats: list[float] = []
	for prop in properties:
		try:
			vals = graycoprops(matrix, prop)
		except Exception:
			vals = np.zeros_like(matrix[..., 0])
		feats.extend(np.asarray(vals).ravel().tolist())

	return np.asarray(feats, dtype=np.float32)


def extract_feature_vector_with_texture(image: np.ndarray, *, use_orb: bool = False, use_glcm: bool = False) -> np.ndarray:
	"""Build a feature vector combining HOG, optional ORB and optional GLCM texture features."""

	base = extract_feature_vector(image, use_orb=use_orb)
	parts: list[np.ndarray] = [base]
	if use_glcm:
		glcm = extract_glcm_features(image)
		parts.append(glcm)
	return np.concatenate(parts).astype(np.float32)


def build_feature_matrix_with_texture(images: list[np.ndarray], *, use_orb: bool = False, use_glcm: bool = False) -> np.ndarray:
	rows = [extract_feature_vector_with_texture(img, use_orb=use_orb, use_glcm=use_glcm) for img in images]
	return np.vstack(rows).astype(np.float32)
