from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..features import geometric, motion, texture
from ..types import FaceRegion, FeatureVector, Landmarks


@dataclass
class FeatureExtractorConfig:
    use_geometric: bool = True
    use_texture: bool = True
    use_deep: bool = False       # disabled by default (requires torch extra)
    use_motion: bool = True
    deep_model: Literal["facenet", "resnet50", "efficientnet_b0"] = "facenet"
    deep_dim: int = 128
    device: str = "cpu"


class FeatureExtractor:
    def __init__(self, config: FeatureExtractorConfig | None = None) -> None:
        self.config = config or FeatureExtractorConfig()
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from ..features.deep import DeepEmbedder
            self._embedder = DeepEmbedder(
                model_name=self.config.deep_model,
                dim=self.config.deep_dim,
                device=self.config.device,
            )
        return self._embedder

    def extract(self, landmarks_seq: list[Landmarks]) -> FeatureVector:
        cfg = self.config
        faces: list[FaceRegion] = [lm.face_region for lm in landmarks_seq]
        images: list[np.ndarray] = [
            f.aligned if f.aligned is not None else f.frame.image for f in faces
        ]
        pts_seq: list[np.ndarray] = [lm.points for lm in landmarks_seq]

        geo = geometric.extract_all(pts_seq) if cfg.use_geometric else np.zeros(0, dtype=np.float32)
        tex = texture.extract_all(images) if cfg.use_texture else np.zeros(0, dtype=np.float32)
        mot = motion.extract_all(images) if cfg.use_motion else np.zeros(0, dtype=np.float32)

        if cfg.use_deep:
            emb = self._get_embedder().embed_sequence_mean(images)
        else:
            emb = np.zeros(0, dtype=np.float32)

        return FeatureVector(
            geometric=geo,
            texture=tex,
            deep=emb,
            motion=mot,
            combined=np.array([], dtype=np.float32),  # filled in __post_init__
            metadata={"n_frames": len(landmarks_seq)},
        )
