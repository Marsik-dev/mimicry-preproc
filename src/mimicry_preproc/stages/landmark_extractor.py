"""
Landmark extraction with MediaPipe → 300W 68-point mapping.

MediaPipe Face Mesh produces 468 landmarks. We map a canonical subset
to the 68-point 300W convention so that all downstream geometric
feature computations use stable, documented indices.

Index mapping source: mediapipe/python/solutions/face_mesh_connections.py
and 300W landmark definitions (https://ibug.doc.ic.ac.uk/resources/300-W/).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from ..types import FaceRegion, Landmarks

# fmt: off
# MediaPipe index → 300W index (0-based).
# Each entry: MP_idx → 300W position in the 68-point set.
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

# Inverse: 300W index → MediaPipe index
W300_TO_MP: dict[int, int] = {v: k for k, v in MP_TO_300W.items()}


@dataclass
class LandmarkExtractorConfig:
    backend: Literal["mediapipe"] = "mediapipe"
    refine: bool = True
    min_confidence: float = 0.5


class LandmarkExtractor:
    def __init__(self, config: LandmarkExtractorConfig | None = None) -> None:
        self.config = config or LandmarkExtractorConfig()
        self._mesh = None
        self._init()

    def _init(self) -> None:
        import mediapipe as mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=self.config.refine,
            min_detection_confidence=self.config.min_confidence,
            min_tracking_confidence=self.config.min_confidence,
        )

    def extract(self, face: FaceRegion) -> Landmarks | None:
        img = face.aligned if face.aligned is not None else face.frame.image
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self._mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return None

        mp_lm = results.multi_face_landmarks[0].landmark
        h, w = img_rgb.shape[:2]

        # Build 68-point array using the canonical MP→300W mapping
        points = np.zeros((68, 2), dtype=np.float32)
        visibility = np.zeros(68, dtype=np.float32)

        for w300_idx in range(68):
            mp_idx = W300_TO_MP.get(w300_idx)
            if mp_idx is not None and mp_idx < len(mp_lm):
                lm = mp_lm[mp_idx]
                points[w300_idx] = [lm.x, lm.y]   # normalized [0, 1]
                visibility[w300_idx] = max(0.0, 1.0 - abs(lm.z))

        return Landmarks(points=points, face_region=face, visibility=visibility)

    def extract_sequence(self, faces: list[FaceRegion]) -> list[Landmarks]:
        results = []
        for f in faces:
            lm = self.extract(f)
            if lm is not None:
                results.append(lm)
        return results

    def close(self) -> None:
        if self._mesh:
            self._mesh.close()

    def __del__(self) -> None:
        self.close()
