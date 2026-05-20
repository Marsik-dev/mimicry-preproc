from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..types import Frame, VideoMetadata


@dataclass
class VideoReaderConfig:
    target_fps: float = 25.0
    max_duration_sec: float = 10.0
    resize_to: tuple[int, int] | None = (227, 227)
    grayscale: bool = True


class VideoReader:
    def __init__(self, config: VideoReaderConfig | None = None) -> None:
        self.config = config or VideoReaderConfig()

    def get_metadata(self, path: str | Path) -> VideoMetadata:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return VideoMetadata(
            path=str(path), fps=fps, width=w, height=h,
            n_frames=n, duration_sec=n / fps if fps else 0.0,
        )

    def read(self, path: str | Path) -> Iterator[Frame]:
        cfg = self.config
        cap = cv2.VideoCapture(str(path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, round(src_fps / cfg.target_fps))
        max_frames = int(cfg.max_duration_sec * cfg.target_fps)

        idx = 0
        out_idx = 0
        while cap.isOpened() and out_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                if cfg.grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if cfg.resize_to:
                    frame = cv2.resize(frame, cfg.resize_to)
                ts = (idx / src_fps) * 1000.0
                yield Frame(image=frame, timestamp_ms=ts, index=out_idx)
                out_idx += 1
            idx += 1
        cap.release()

    def read_frames(self, path: str | Path, max_frames: int | None = None) -> list[Frame]:
        frames = list(self.read(path))
        return frames[:max_frames] if max_frames else frames

    def read_from_array(self, frames_bgr: list[np.ndarray], fps: float = 25.0) -> list[Frame]:
        """Accept pre-captured frames (e.g. from webcam buffer)."""
        cfg = self.config
        result = []
        for i, frame in enumerate(frames_bgr):
            if cfg.grayscale and frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if cfg.resize_to:
                frame = cv2.resize(frame, cfg.resize_to)
            result.append(Frame(image=frame, timestamp_ms=i * 1000.0 / fps, index=i))
        return result
