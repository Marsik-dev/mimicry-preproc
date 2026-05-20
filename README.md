# mimicry-preproc — Facial Expression Video Preprocessing Pipeline

Composable video preprocessing pipeline that converts raw facial expression video into a feature vector suitable for NPBK biometric training.

## Pipeline

```
VideoReader → FaceDetector → QualityFilter → Stabilizer →
LandmarkExtractor → FeatureExtractor → Normalizer → FeatureVector
```

**Features extracted:**
- **Geometric** — scale-invariant distances, angles, triangle area ratios (68-point 300W landmarks)
- **Texture** — LBP, HOG, Gabor filter responses
- **Motion** — optical flow statistics (mean, variance, skewness, kurtosis)
- **Deep** *(optional)* — FaceNet/ResNet-50/EfficientNet embeddings

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Optional: deep embeddings (requires PyTorch ~2 GB)
pip install -e ".[dev,deep]"
```

## MediaPipe models

Models are bundled in `models/` (downloaded automatically by the workflow):
- `models/face_detector.tflite` — BlazeFace short-range
- `models/face_landmarker.task` — FaceLandmarker

## Usage

```python
from mimicry_preproc import Pipeline

pipeline = Pipeline()
fv = pipeline.run("emotion_video.mp4")
print(fv.combined.shape)  # e.g. (6180,)
```

## Tests

```bash
pytest tests/ -v
```
