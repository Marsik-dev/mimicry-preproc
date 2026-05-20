import numpy as np
import pytest

from mimicry_preproc.features.geometric import extract_all, extract_from_frame


def test_extract_from_frame_shape():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, (68, 2)).astype(np.float32)
    feat = extract_from_frame(pts)
    assert feat.ndim == 1
    assert feat.dtype == np.float32
    assert len(feat) > 0


def test_extract_all_shape():
    rng = np.random.default_rng(1)
    seq = [rng.uniform(0, 1, (68, 2)).astype(np.float32) for _ in range(10)]
    feat = extract_all(seq)
    single = extract_from_frame(seq[0])
    assert feat.shape == (2 * len(single),)


def test_extract_all_empty():
    feat = extract_all([])
    assert feat.ndim == 1
    assert feat.size > 0  # returns zeros of correct size


def test_scale_invariance():
    rng = np.random.default_rng(2)
    pts = rng.uniform(0.1, 0.9, (68, 2)).astype(np.float32)
    feat1 = extract_from_frame(pts)
    feat2 = extract_from_frame(pts * 2.0)  # scale by 2
    np.testing.assert_allclose(feat1, feat2, rtol=1e-4)
