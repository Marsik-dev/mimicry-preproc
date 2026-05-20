"""Main Pipeline orchestrator for video → FeatureVector."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .stages.face_detector import FaceDetector, FaceDetectorConfig
from .stages.feature_extractor import FeatureExtractor, FeatureExtractorConfig
from .stages.landmark_extractor import LandmarkExtractor, LandmarkExtractorConfig
from .stages.normalizer import Normalizer, NormalizerConfig
from .stages.quality_filter import QualityFilter, QualityFilterConfig
from .stages.stabilizer import Stabilizer, StabilizerConfig
from .stages.video_reader import VideoReader, VideoReaderConfig
from .types import FeatureVector, Frame, Landmarks


@dataclass
class PipelineConfig:
    video_reader: VideoReaderConfig = field(default_factory=VideoReaderConfig)
    face_detector: FaceDetectorConfig = field(default_factory=FaceDetectorConfig)
    quality_filter: QualityFilterConfig = field(default_factory=QualityFilterConfig)
    stabilizer: StabilizerConfig = field(default_factory=StabilizerConfig)
    landmark_extractor: LandmarkExtractorConfig = field(default_factory=LandmarkExtractorConfig)
    feature_extractor: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)

    @classmethod
    def default(cls) -> "PipelineConfig":
        return cls()

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineConfig":
        return cls(
            video_reader=VideoReaderConfig(**d.get("video_reader", {})),
            face_detector=FaceDetectorConfig(**d.get("face_detector", {})),
            quality_filter=QualityFilterConfig(**d.get("quality_filter", {})),
            stabilizer=StabilizerConfig(**d.get("stabilizer", {})),
            landmark_extractor=LandmarkExtractorConfig(**d.get("landmark_extractor", {})),
            feature_extractor=FeatureExtractorConfig(**d.get("feature_extractor", {})),
            normalizer=NormalizerConfig(**d.get("normalizer", {})),
        )


@dataclass
class StageOutput:
    """Intermediate results from each pipeline stage for visualization."""
    frames: list[Frame] | None = None
    faces: list | None = None         # list[FaceRegion | None]
    faces_filtered: list | None = None
    faces_stabilized: list | None = None
    landmarks: list | None = None     # list[Landmarks]
    feature_vector: FeatureVector | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


class Pipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig.default()
        self._video_reader = VideoReader(self.config.video_reader)
        self._face_detector = FaceDetector(self.config.face_detector)
        self._quality_filter = QualityFilter(self.config.quality_filter)
        self._stabilizer = Stabilizer(self.config.stabilizer)
        self._landmark_extractor = LandmarkExtractor(self.config.landmark_extractor)
        self._feature_extractor = FeatureExtractor(self.config.feature_extractor)
        self._normalizer: Normalizer | None = None  # optional, fitted separately

    def run(self, video_path: str | Path) -> FeatureVector:
        return self._run_internal(video_path).feature_vector  # type: ignore[return-value]

    def run_with_debug(self, video_path: str | Path) -> StageOutput:
        return self._run_internal(video_path)

    def run_with_debug_from_frames(self, frames_bgr: list[np.ndarray], fps: float = 25.0) -> StageOutput:
        frames = self._video_reader.read_from_array(frames_bgr, fps)
        return self._process_frames(frames)

    def run_from_frames(self, frames_bgr: list[np.ndarray], fps: float = 25.0) -> FeatureVector:
        frames = self._video_reader.read_from_array(frames_bgr, fps)
        return self._process_frames(frames).feature_vector  # type: ignore[return-value]

    def _run_internal(self, video_path: str | Path) -> StageOutput:
        import time
        out = StageOutput()
        t0 = time.perf_counter()

        frames = self._video_reader.read_frames(video_path)
        out.frames = frames
        out.timings_ms["video_reader"] = (time.perf_counter() - t0) * 1000

        return self._process_frames_from_stage(out)

    def _process_frames(self, frames: list[Frame]) -> StageOutput:
        out = StageOutput(frames=frames)
        return self._process_frames_from_stage(out)

    def _process_frames_from_stage(self, out: StageOutput) -> StageOutput:
        import time

        frames = out.frames or []

        t = time.perf_counter()
        faces_raw = self._face_detector.detect_batch(frames)
        faces = [f for f in faces_raw if f is not None]
        out.faces = faces_raw
        out.timings_ms["face_detector"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        faces_filtered = self._quality_filter.filter(faces)
        out.faces_filtered = faces_filtered
        out.timings_ms["quality_filter"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        faces_stable = self._stabilizer.stabilize(faces_filtered)
        out.faces_stabilized = faces_stable
        out.timings_ms["stabilizer"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        landmarks = self._landmark_extractor.extract_sequence(faces_stable)
        out.landmarks = landmarks
        out.timings_ms["landmark_extractor"] = (time.perf_counter() - t) * 1000

        if not landmarks:
            out.feature_vector = FeatureVector(
                geometric=np.zeros(0, dtype=np.float32),
                texture=np.zeros(0, dtype=np.float32),
                deep=np.zeros(0, dtype=np.float32),
                motion=np.zeros(0, dtype=np.float32),
                combined=np.zeros(0, dtype=np.float32),
                metadata={"error": "no_landmarks_detected"},
            )
            return out

        t = time.perf_counter()
        fv = self._feature_extractor.extract(landmarks)
        out.timings_ms["feature_extractor"] = (time.perf_counter() - t) * 1000

        if self._normalizer is not None:
            t = time.perf_counter()
            fv = self._normalizer.transform(fv)
            out.timings_ms["normalizer"] = (time.perf_counter() - t) * 1000

        out.feature_vector = fv
        return out

    def set_normalizer(self, normalizer: Normalizer) -> None:
        self._normalizer = normalizer
