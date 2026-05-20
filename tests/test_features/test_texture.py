import numpy as np

from mimicry_preproc.features.texture import extract_all, gabor_response, hog_descriptor, lbp_histogram


def _make_img(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (64, 64), dtype=np.uint8)


def test_lbp_histogram_shape():
    feat = lbp_histogram(_make_img())
    assert feat.ndim == 1
    assert len(feat) == 10  # n_points=8 → 8+2 bins


def test_hog_descriptor_shape():
    feat = hog_descriptor(_make_img())
    assert feat.ndim == 1
    assert len(feat) > 0


def test_gabor_response_shape():
    feat = gabor_response(_make_img())
    assert feat.ndim == 1
    assert len(feat) == 3 * 4 * 2  # 3 freqs × 4 thetas × 2 stats


def test_extract_all_shape():
    imgs = [_make_img(i) for i in range(5)]
    feat = extract_all(imgs)
    single = np.concatenate([lbp_histogram(imgs[0]), hog_descriptor(imgs[0]), gabor_response(imgs[0])])
    assert feat.shape == single.shape
