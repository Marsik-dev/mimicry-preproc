"""Optical flow motion features across a video sequence."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import kurtosis, skew

from ..types import OpticalFlowField


def compute_flow_sequence(frames: list[np.ndarray]) -> list[OpticalFlowField]:
    """Compute dense optical flow between consecutive frames."""
    if len(frames) < 2:
        return []
    results = []
    for prev, curr in zip(frames[:-1], frames[1:]):
        g_prev = prev if prev.ndim == 2 else cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        g_curr = curr if curr.ndim == 2 else cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            g_prev, g_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        results.append(OpticalFlowField(flow=flow, magnitude=mag, angle=ang))
    return results


def flow_statistics(flow_fields: list[OpticalFlowField]) -> np.ndarray:
    """
    Aggregate optical flow into a fixed-size feature vector.
    Returns: [mean_mag, std_mag, skew_mag, kurt_mag,
               mean_ang, std_ang, skew_ang, kurt_ang,
               mean_dx, std_dx, mean_dy, std_dy]
    """
    if not flow_fields:
        return np.zeros(12, dtype=np.float32)

    mags = np.concatenate([f.magnitude.ravel() for f in flow_fields])
    angs = np.concatenate([f.angle.ravel() for f in flow_fields])
    dx = np.concatenate([f.flow[..., 0].ravel() for f in flow_fields])
    dy = np.concatenate([f.flow[..., 1].ravel() for f in flow_fields])

    feats = [
        mags.mean(), mags.std(),
        float(skew(mags)), float(kurtosis(mags)),
        angs.mean(), angs.std(),
        float(skew(angs)), float(kurtosis(angs)),
        dx.mean(), dx.std(),
        dy.mean(), dy.std(),
    ]
    return np.array(feats, dtype=np.float32)


def extract_all(face_images: list[np.ndarray]) -> np.ndarray:
    """Convenience: compute flow sequence → statistics in one call."""
    flow_fields = compute_flow_sequence(face_images)
    return flow_statistics(flow_fields)
