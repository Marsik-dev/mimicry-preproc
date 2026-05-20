from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from ..types import FaceRegion, OpticalFlowField


@dataclass
class StabilizerConfig:
    method: Literal["affine", "optical_flow"] = "affine"
    reference_frame: Literal["first", "median_idx"] = "first"


class Stabilizer:
    def __init__(self, config: StabilizerConfig | None = None) -> None:
        self.config = config or StabilizerConfig()

    def stabilize(self, faces: list[FaceRegion]) -> list[FaceRegion]:
        if len(faces) < 2:
            return faces
        ref_img = self._get_reference(faces)
        return [self._warp_to_reference(f, ref_img) for f in faces]

    def _get_reference(self, faces: list[FaceRegion]) -> np.ndarray:
        f = faces[0] if self.config.reference_frame == "first" else faces[len(faces) // 2]
        img = f.aligned if f.aligned is not None else f.frame.image
        return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _warp_to_reference(self, face: FaceRegion, ref_gray: np.ndarray) -> FaceRegion:
        img = face.aligned if face.aligned is not None else face.frame.image
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.config.method == "affine":
            try:
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
                _, warp_matrix = cv2.findTransformECC(
                    ref_gray, gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
                )
                h, w = gray.shape[:2]
                stabilized = cv2.warpAffine(img, warp_matrix, (w, h),
                                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            except cv2.error:
                stabilized = img
        else:
            stabilized = img

        return FaceRegion(
            frame=face.frame,
            bbox=face.bbox,
            confidence=face.confidence,
            aligned=stabilized,
        )

    def compute_flow(self, face_a: FaceRegion, face_b: FaceRegion) -> OpticalFlowField:
        def to_gray(f: FaceRegion) -> np.ndarray:
            img = f.aligned if f.aligned is not None else f.frame.image
            return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        g_a = to_gray(face_a)
        g_b = to_gray(face_b)
        flow = cv2.calcOpticalFlowFarneback(g_a, g_b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return OpticalFlowField(flow=flow, magnitude=magnitude, angle=angle)
