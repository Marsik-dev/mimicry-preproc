"""Texture features: LBP, HOG, Gabor filters."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


def lbp_histogram(image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def hog_descriptor(
    image: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (16, 16),
    cells_per_block: tuple[int, int] = (2, 2),
) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feat = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        feature_vector=True,
    )
    return feat.astype(np.float32)


def gabor_response(
    image: np.ndarray,
    frequencies: list[float] | None = None,
    thetas: list[float] | None = None,
) -> np.ndarray:
    if frequencies is None:
        frequencies = [0.1, 0.2, 0.3]
    if thetas is None:
        thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    responses: list[float] = []
    for freq in frequencies:
        for theta in thetas:
            kernel = cv2.getGaborKernel(
                (21, 21), sigma=4.0, theta=theta,
                lambd=1.0 / freq, gamma=0.5, psi=0,
            )
            filtered = cv2.filter2D(gray_f, cv2.CV_32F, kernel)
            responses.extend([float(filtered.mean()), float(filtered.var())])
    return np.array(responses, dtype=np.float32)


def extract_all(images: list[np.ndarray]) -> np.ndarray:
    """Compute mean LBP + HOG + Gabor over a sequence of face images."""
    if not images:
        return np.zeros(0, dtype=np.float32)
    lbp_feats = np.stack([lbp_histogram(img) for img in images]).mean(axis=0)
    hog_feats = np.stack([hog_descriptor(img) for img in images]).mean(axis=0)
    gabor_feats = np.stack([gabor_response(img) for img in images]).mean(axis=0)
    return np.concatenate([lbp_feats, hog_feats, gabor_feats]).astype(np.float32)
