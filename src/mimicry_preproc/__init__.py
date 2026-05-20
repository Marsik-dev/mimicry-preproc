"""mimicry-preproc — video preprocessing pipeline for facial expression biometrics."""
from .pipeline import Pipeline, PipelineConfig, StageOutput
from .types import FaceRegion, FeatureVector, Frame, Landmarks, OpticalFlowField

__version__ = "0.1.0"
__all__ = [
    "Pipeline",
    "PipelineConfig",
    "StageOutput",
    "Frame",
    "FaceRegion",
    "Landmarks",
    "FeatureVector",
    "OpticalFlowField",
]
