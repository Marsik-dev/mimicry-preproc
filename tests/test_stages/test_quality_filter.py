import numpy as np
import pytest

from mimicry_preproc.stages.quality_filter import QualityFilter, QualityFilterConfig
from mimicry_preproc.types import FaceRegion, Frame


def _make_face(brightness: int = 128, blur: bool = False) -> FaceRegion:
    img = np.full((64, 64), brightness, dtype=np.uint8)
    if not blur:
        # Add edges to increase sharpness
        img[32, :] = 255
        img[:, 32] = 0
    frame = Frame(image=img, timestamp_ms=0, index=0)
    return FaceRegion(frame=frame, bbox=(0, 0, 64, 64), confidence=0.9, aligned=img)


def test_sharp_bright_face_passes():
    face = _make_face(brightness=128, blur=False)
    qf = QualityFilter(QualityFilterConfig(min_sharpness=1.0))
    score = qf.assess(face)
    assert score.passed


def test_dark_face_fails():
    face = _make_face(brightness=5, blur=True)
    qf = QualityFilter(QualityFilterConfig(min_brightness=0.1))
    score = qf.assess(face)
    assert not score.passed


def test_filter_removes_bad():
    good = _make_face(128, False)
    bad = _make_face(5, True)
    qf = QualityFilter(QualityFilterConfig(min_sharpness=1.0, min_brightness=0.1))
    kept = qf.filter([good, bad])
    assert len(kept) == 1
