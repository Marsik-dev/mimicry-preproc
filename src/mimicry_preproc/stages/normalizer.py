from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from ..types import FeatureVector


@dataclass
class NormalizerConfig:
    method: Literal["standard", "minmax"] = "standard"
    reduce: Literal["pca", "lda", "none"] = "pca"
    n_components: int | float = 0.95   # variance ratio for PCA, or int for fixed


class Normalizer:
    def __init__(self, config: NormalizerConfig | None = None) -> None:
        self.config = config or NormalizerConfig()
        self._scaler = StandardScaler()
        self._reducer = None
        self._fitted = False

    def fit(self, vectors: list[FeatureVector], labels: list[int] | None = None) -> None:
        X = np.stack([v.combined for v in vectors])
        self._scaler.fit(X)
        Xs = self._scaler.transform(X)

        cfg = self.config
        if cfg.reduce == "pca":
            self._reducer = PCA(n_components=cfg.n_components, whiten=True)
            self._reducer.fit(Xs)
        elif cfg.reduce == "lda" and labels is not None:
            self._reducer = LinearDiscriminantAnalysis(n_components=None)
            self._reducer.fit(Xs, labels)
        self._fitted = True

    def transform(self, vector: FeatureVector) -> FeatureVector:
        x = vector.combined.reshape(1, -1)
        x = self._scaler.transform(x)
        if self._reducer is not None:
            x = self._reducer.transform(x)
        combined = x.squeeze(0).astype(np.float32)
        return FeatureVector(
            geometric=vector.geometric,
            texture=vector.texture,
            deep=vector.deep,
            motion=vector.motion,
            combined=combined,
            metadata=vector.metadata,
        )

    def fit_transform(
        self, vectors: list[FeatureVector], labels: list[int] | None = None
    ) -> list[FeatureVector]:
        self.fit(vectors, labels)
        return [self.transform(v) for v in vectors]

    def save(self, path: str | Path) -> None:
        import pickle
        Path(path).write_bytes(pickle.dumps({"scaler": self._scaler, "reducer": self._reducer, "fitted": self._fitted, "config": self.config}))

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        import pickle
        data = pickle.loads(Path(path).read_bytes())
        obj = cls(data["config"])
        obj._scaler = data["scaler"]
        obj._reducer = data["reducer"]
        obj._fitted = data["fitted"]
        return obj
