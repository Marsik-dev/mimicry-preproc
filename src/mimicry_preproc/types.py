from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Frame:
    image: np.ndarray      # (H, W) grayscale uint8 or (H, W, 3) BGR uint8
    timestamp_ms: float
    index: int


@dataclass
class FaceRegion:
    frame: Frame
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    aligned: np.ndarray | None = None  # aligned crop (H, W) or (H, W, 3)


@dataclass
class Landmarks:
    """68 facial keypoints in 300W convention."""
    points: np.ndarray       # (68, 2) float32, normalized to [0, 1] within bbox
    face_region: FaceRegion
    visibility: np.ndarray   # (68,) float32, per-point confidence


@dataclass
class OpticalFlowField:
    flow: np.ndarray      # (H, W, 2) float32
    magnitude: np.ndarray  # (H, W) float32
    angle: np.ndarray      # (H, W) float32


@dataclass
class FeatureVector:
    geometric: np.ndarray   # scale-invariant geometric features
    texture: np.ndarray     # LBP + HOG + Gabor concatenated
    deep: np.ndarray        # deep embedding (128–512 dim); empty if not computed
    motion: np.ndarray      # optical flow statistics
    combined: np.ndarray    # final concatenated + normalized vector → passed to npbk
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.combined.size == 0:
            parts = [p for p in (self.geometric, self.texture, self.deep, self.motion) if p.size]
            self.combined = np.concatenate(parts) if parts else np.array([], dtype=np.float32)


@dataclass
class VideoMetadata:
    path: str
    fps: float
    width: int
    height: int
    n_frames: int
    duration_sec: float
