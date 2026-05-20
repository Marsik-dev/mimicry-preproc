from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from ..types import FaceRegion, Frame


@dataclass
class FaceDetectorConfig:
    backend: Literal["mediapipe", "opencv_haar"] = "mediapipe"
    min_confidence: float = 0.5
    output_size: tuple[int, int] = (227, 227)
    align: bool = True


class FaceDetector:
    def __init__(self, config: FaceDetectorConfig | None = None) -> None:
        self.config = config or FaceDetectorConfig()
        self._mp_face = None
        self._haar = None
        self._init_backend()

    def _init_backend(self) -> None:
        if self.config.backend == "mediapipe":
            import mediapipe as mp
            self._mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=self.config.min_confidence,
            )

    def detect(self, frame: Frame) -> FaceRegion | None:
        img = frame.image
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.config.backend == "mediapipe" and self._mp_face:
            results = self._mp_face.process(img_rgb)
            if not results.detections:
                return None
            det = results.detections[0]
            score = det.score[0]
            if score < self.config.min_confidence:
                return None
            h, w = img.shape[:2]
            bb = det.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * w))
            y = max(0, int(bb.ymin * h))
            bw = int(bb.width * w)
            bh = int(bb.height * h)
            aligned = self._crop_and_align(img, x, y, bw, bh)
            return FaceRegion(frame=frame, bbox=(x, y, bw, bh), confidence=float(score), aligned=aligned)

        return None

    def _crop_and_align(
        self, img: np.ndarray, x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        x2 = min(img.shape[1], x + w)
        y2 = min(img.shape[0], y + h)
        crop = img[y:y2, x:x2]
        if crop.size == 0:
            return np.zeros((*self.config.output_size, ) if img.ndim == 2 else (*self.config.output_size, 3), dtype=np.uint8)
        return cv2.resize(crop, self.config.output_size)

    def detect_batch(self, frames: list[Frame]) -> list[FaceRegion | None]:
        return [self.detect(f) for f in frames]

    def close(self) -> None:
        if self._mp_face:
            self._mp_face.close()

    def __del__(self) -> None:
        self.close()
