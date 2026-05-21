from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from ..types import FaceRegion, Frame

_DEFAULT_MODEL = Path(__file__).parent.parent / "models" / "face_detector.tflite"


@dataclass
class FaceDetectorConfig:
    backend: Literal["mediapipe", "opencv_haar"] = "mediapipe"
    min_confidence: float = 0.5
    output_size: tuple[int, int] = (227, 227)
    align: bool = True
    model_path: str | None = None  # None = use bundled model


class FaceDetector:
    def __init__(self, config: FaceDetectorConfig | None = None) -> None:
        self.config = config or FaceDetectorConfig()
        self._detector = None
        self._init_backend()

    def _init_backend(self) -> None:
        if self.config.backend == "mediapipe":
            import mediapipe as mp
            BaseOptions = mp.tasks.BaseOptions
            FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
            RunningMode = mp.tasks.vision.RunningMode

            model_path = self.config.model_path or str(_DEFAULT_MODEL)
            options = FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                min_detection_confidence=self.config.min_confidence,
            )
            self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    def detect(self, frame: Frame) -> FaceRegion | None:
        img = frame.image
        if self._detector is None:
            return None

        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self._detector.detect(mp_image)

        if not result.detections:
            return None

        det = result.detections[0]
        score = det.categories[0].score if det.categories else 0.0
        if score < self.config.min_confidence:
            return None

        h, w = img.shape[:2]
        bb = det.bounding_box
        x = max(0, bb.origin_x)
        y = max(0, bb.origin_y)
        bw = bb.width
        bh = bb.height

        aligned = self._crop_and_align(img, x, y, bw, bh)
        return FaceRegion(frame=frame, bbox=(x, y, bw, bh), confidence=float(score), aligned=aligned)

    def _crop_and_align(
        self, img: np.ndarray, x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        x2 = min(img.shape[1], x + w)
        y2 = min(img.shape[0], y + h)
        crop = img[y:y2, x:x2]
        if crop.size == 0:
            sz = self.config.output_size
            shape = sz if img.ndim == 2 else (*sz, 3)
            return np.zeros(shape, dtype=np.uint8)
        return cv2.resize(crop, self.config.output_size)

    def detect_batch(self, frames: list[Frame]) -> list[FaceRegion | None]:
        return [self.detect(f) for f in frames]

    def close(self) -> None:
        if self._detector:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
