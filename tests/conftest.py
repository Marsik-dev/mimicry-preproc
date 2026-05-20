import numpy as np
import pytest

from mimicry_preproc.types import FaceRegion, Frame, Landmarks


@pytest.fixture
def dummy_frame() -> Frame:
    img = np.random.randint(100, 200, (227, 227), dtype=np.uint8)
    return Frame(image=img, timestamp_ms=0.0, index=0)


@pytest.fixture
def dummy_face(dummy_frame: Frame) -> FaceRegion:
    aligned = np.random.randint(50, 200, (227, 227), dtype=np.uint8)
    return FaceRegion(frame=dummy_frame, bbox=(10, 10, 100, 100), confidence=0.95, aligned=aligned)


@pytest.fixture
def dummy_landmarks(dummy_face: FaceRegion) -> Landmarks:
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.1, 0.9, (68, 2)).astype(np.float32)
    vis = np.ones(68, dtype=np.float32)
    return Landmarks(points=pts, face_region=dummy_face, visibility=vis)
