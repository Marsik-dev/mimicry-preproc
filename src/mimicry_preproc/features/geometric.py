"""
Scale-invariant geometric features from 68-point landmarks (300W convention).

All distances are normalized by the inter-ocular distance (IOD) so that
features are scale-invariant.
"""
from __future__ import annotations

import numpy as np

# --- Index groups (300W 0-based) ---
# Jawline: 0–16,  R-brow: 17–21,  L-brow: 22–26
# Nose bridge: 27–30,  Nose base: 31–35
# R-eye: 36–41,  L-eye: 42–47
# Outer lips: 48–59,  Inner lips: 60–67

_DISTANCE_PAIRS: list[tuple[int, int]] = [
    # Eye geometry
    (36, 39), (42, 45),   # eye widths
    (37, 41), (43, 47),   # eye heights
    (36, 42),             # inter-ocular (also used for normalization)
    # Eyebrow position
    (19, 37), (24, 43),   # brow-to-eye vertical
    # Nose
    (27, 33), (31, 35), (33, 30),
    # Mouth
    (48, 54),             # mouth width
    (51, 57),             # mouth height
    (61, 67), (62, 66),   # inner lip heights
    # Chin
    (8, 57),              # chin to mouth center
    (0, 16),              # jaw width
    # Cheek-to-eye
    (1, 36), (15, 45),
]

_ANGLE_TRIPLETS: list[tuple[int, int, int]] = [
    # (vertex, p1, p2) — angle at vertex
    (17, 19, 22),  # brow tilt
    (48, 51, 54),  # mouth corner angle
    (36, 38, 39),  # eye corner angle R
    (42, 44, 45),  # eye corner angle L
]

_TRIANGLE_AREAS: list[tuple[tuple[int, int, int], tuple[int, int]]] = [
    # (triangle_points, reference_pair for normalization)
    ((36, 39, 27), (36, 42)),   # eye to nose bridge
    ((48, 54, 57), (48, 54)),   # mouth triangle
    ((19, 24, 28), (36, 42)),   # brow to nose
]


def _iod(pts: np.ndarray) -> float:
    """Inter-ocular distance: distance between outer eye corners."""
    return float(np.linalg.norm(pts[36] - pts[45])) + 1e-8


def _dist(pts: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(pts[i] - pts[j]))


def _angle_at_vertex(pts: np.ndarray, v: int, p1: int, p2: int) -> float:
    a = pts[p1] - pts[v]
    b = pts[p2] - pts[v]
    cos_a = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def _triangle_area(pts: np.ndarray, i: int, j: int, k: int) -> float:
    a, b, c = pts[i], pts[j], pts[k]
    return float(abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0)


def extract_from_frame(pts: np.ndarray) -> np.ndarray:
    """
    pts: (68, 2) landmarks in image coordinates.
    Returns scale-invariant feature vector.
    """
    iod = _iod(pts)
    feats: list[float] = []

    for i, j in _DISTANCE_PAIRS:
        feats.append(_dist(pts, i, j) / iod)

    for v, p1, p2 in _ANGLE_TRIPLETS:
        feats.append(_angle_at_vertex(pts, v, p1, p2))

    for (ti, tj, tk), (ri, rj) in _TRIANGLE_AREAS:
        ref = _dist(pts, ri, rj) ** 2 + 1e-8
        feats.append(_triangle_area(pts, ti, tj, tk) / ref)

    return np.array(feats, dtype=np.float32)


def extract_all(landmarks_seq: list[np.ndarray]) -> np.ndarray:
    """
    landmarks_seq: list of (68, 2) arrays across video frames.
    Returns mean + std of per-frame features → shape (2 * n_features,).
    """
    if not landmarks_seq:
        n = len(_DISTANCE_PAIRS) + len(_ANGLE_TRIPLETS) + len(_TRIANGLE_AREAS)
        return np.zeros(2 * n, dtype=np.float32)
    frames = np.stack([extract_from_frame(p) for p in landmarks_seq])  # (T, F)
    return np.concatenate([frames.mean(axis=0), frames.std(axis=0)]).astype(np.float32)
