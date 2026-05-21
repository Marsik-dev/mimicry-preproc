"""
Landmark extraction with MediaPipe Tasks → 300W 68-point mapping.

MediaPipe FaceLandmarker (Tasks API, ≥0.10) produces 478 landmarks.
We map a canonical subset to the 68-point 300W convention so that all
downstream geometric feature computations use stable, documented indices.

Index mapping source:
  - MediaPipe face_landmarker landmark indices
  - 300W landmark definitions (https://ibug.doc.ic.ac.uk/resources/300-W/)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from ..types import FaceRegion, Landmarks

_DEFAULT_MODEL = Path(__file__).parent.parent / "models" / "face_landmarker.task"

# fmt: off
# MediaPipe FaceLandmarker index → 300W index (0-based).
MP_TO_300W: dict[int, int] = {
    # Jawline (0–16)
    127: 0,  234: 1,  93: 2,   132: 3,  58: 4,   172: 5,
    136: 6,  150: 7,  149: 8,  176: 9,  148: 10, 152: 11,
    377: 12, 400: 13, 378: 14, 379: 15, 365: 16,
    # Right eyebrow (17–21)
    70:  17, 63:  18, 105: 19, 66:  20, 107: 21,
    # Left eyebrow (22–26)
    336: 22, 296: 23, 334: 24, 293: 25, 300: 26,
    # Nose bridge (27–30)
    168: 27, 6:   28, 197: 29, 195: 30,
    # Nose base (31–35)
    4:   31, 240: 32, 97:  33, 2:   34, 326: 35,
    # Right eye (36–41)
    33:  36, 160: 37, 158: 38, 133: 39, 153: 40, 144: 41,
    # Left eye (42–47)
    362: 42, 385: 43, 387: 44, 263: 45, 373: 46, 380: 47,
    # Outer lips (48–59)
    61:  48, 39:  49, 37:  50, 0:   51, 267: 52, 269: 53,
    291: 54, 405: 55, 314: 56, 17:  57, 84:  58, 181: 59,
    # Inner lips (60–67)
    78:  60, 82:  61, 13:  62, 312: 63, 308: 64, 317: 65,
    14:  66, 87:  67,
}
# fmt: on

W300_TO_MP: dict[int, int] = {v: k for k, v in MP_TO_300W.items()}


@dataclass
class LandmarkExtractorConfig:
    backend: Literal["mediapipe"] = "mediapipe"
    min_confidence: float = 0.5
    model_path: str | None = None


class LandmarkExtractor:
    def __init__(self, config: LandmarkExtractorConfig | None = None) -> None:
        self.config = config or LandmarkExtractorConfig()
        self._landmarker = None
        self._init()

    def _init(self) -> None:
        import mediapipe as mp
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        model_path = self.config.model_path or str(_DEFAULT_MODEL)
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=self.config.min_confidence,
            min_face_presence_confidence=self.config.min_confidence,
            min_tracking_confidence=self.config.min_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def extract(self, face: FaceRegion) -> Landmarks | None:
        img = face.aligned if face.aligned is not None else face.frame.image
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        mp_lm = result.face_landmarks[0]
        points = np.zeros((68, 2), dtype=np.float32)
        visibility = np.zeros(68, dtype=np.float32)

        for w300_idx in range(68):
            mp_idx = W300_TO_MP.get(w300_idx)
            if mp_idx is not None and mp_idx < len(mp_lm):
                lm = mp_lm[mp_idx]
                points[w300_idx] = [lm.x, lm.y]
                visibility[w300_idx] = max(0.0, 1.0 - abs(lm.z))

        return Landmarks(points=points, face_region=face, visibility=visibility)

    def extract_sequence(self, faces: list[FaceRegion]) -> list[Landmarks]:
        return [lm for f in faces if (lm := self.extract(f)) is not None]

    def close(self) -> None:
        if self._landmarker:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
