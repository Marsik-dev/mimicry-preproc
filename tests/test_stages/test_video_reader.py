import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from mimicry_preproc.stages.video_reader import VideoReader, VideoReaderConfig


def _write_test_video(path: str, n_frames: int = 30, fps: float = 30.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (64, 64))
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        frame = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out.write(frame)
    out.release()


def test_read_from_array():
    rng = np.random.default_rng(0)
    frames_bgr = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(10)]
    reader = VideoReader(VideoReaderConfig(grayscale=True, resize_to=(32, 32)))
    frames = reader.read_from_array(frames_bgr)
    assert len(frames) == 10
    assert frames[0].image.shape == (32, 32)


def test_read_video_file():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        _write_test_video(f.name, n_frames=30, fps=30.0)
        reader = VideoReader(VideoReaderConfig(target_fps=15.0, resize_to=(32, 32)))
        frames = reader.read_frames(f.name)
    assert len(frames) > 0
    assert frames[0].image.shape[:2] == (32, 32)
