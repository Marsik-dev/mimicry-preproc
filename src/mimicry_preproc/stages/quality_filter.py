from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..types import FaceRegion


@dataclass
class QualityScore:
    sharpness: float   # Laplacian variance
    brightness: float  # mean pixel value normalized to [0, 1]
    overall: float
    passed: bool


@dataclass
class QualityFilterConfig:
    min_sharpness: float = 50.0
    min_brightness: float = 0.1
    max_brightness: float = 0.95
    min_overall: float = 0.4


class QualityFilter:
    def __init__(self, config: QualityFilterConfig | None = None) -> None:
        self.config = config or QualityFilterConfig()

    def assess(self, face: FaceRegion) -> QualityScore:
        img = face.aligned if face.aligned is not None else face.frame.image
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean()) / 255.0

        cfg = self.config
        sharp_ok = sharpness >= cfg.min_sharpness
        bright_ok = cfg.min_brightness <= brightness <= cfg.max_brightness
        overall = float(sharp_ok) * 0.6 + float(bright_ok) * 0.4
        passed = sharp_ok and bright_ok and overall >= cfg.min_overall

        return QualityScore(sharpness=sharpness, brightness=brightness, overall=overall, passed=passed)

    def filter(self, faces: list[FaceRegion]) -> list[FaceRegion]:
        return [f for f in faces if self.assess(f).passed]
