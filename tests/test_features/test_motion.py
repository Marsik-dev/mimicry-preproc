import numpy as np

from mimicry_preproc.features.motion import compute_flow_sequence, extract_all, flow_statistics


def _make_frames(n: int = 5, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (64, 64), dtype=np.uint8) for _ in range(n)]


def test_flow_sequence_length():
    frames = _make_frames(5)
    flows = compute_flow_sequence(frames)
    assert len(flows) == 4  # N-1 pairs


def test_flow_statistics_shape():
    frames = _make_frames(5)
    flows = compute_flow_sequence(frames)
    stats = flow_statistics(flows)
    assert stats.shape == (12,)
    assert stats.dtype == np.float32


def test_flow_empty():
    stats = flow_statistics([])
    assert stats.shape == (12,)
    assert (stats == 0).all()


def test_extract_all_shape():
    frames = _make_frames(5)
    feat = extract_all(frames)
    assert feat.shape == (12,)
