"""Deep embedding extraction (optional, requires torch extra)."""
from __future__ import annotations

from typing import Literal

import numpy as np


class DeepEmbedder:
    """
    Extracts deep embeddings from face images using a pretrained model.
    Requires: pip install mimicry-preproc[deep]
    """

    def __init__(
        self,
        model_name: Literal["facenet", "resnet50", "efficientnet_b0"] = "facenet",
        dim: int = 128,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self.device = device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError:
            raise ImportError("Deep embeddings require: pip install mimicry-preproc[deep]")

        if self.model_name == "facenet":
            from facenet_pytorch import InceptionResnetV1
            self._model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            self.dim = 512
        elif self.model_name == "resnet50":
            import torchvision.models as models
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            base.fc = torch.nn.Identity()
            self._model = base.eval().to(self.device)
            self.dim = 2048
        else:
            import torchvision.models as models
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            base.classifier = torch.nn.Identity()
            self._model = base.eval().to(self.device)
            self.dim = 1280

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        """images: list of (H, W) or (H, W, 3) uint8. Returns (N, dim)."""
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image

        self._load()
        tensors = []
        for img in images:
            if img.ndim == 2:
                pil = Image.fromarray(img).convert("RGB")
            else:
                import cv2
                pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            t = TF.to_tensor(TF.resize(pil, [160, 160]))
            t = TF.normalize(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            tensors.append(t)

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            embs = self._model(batch).cpu().numpy()
        return embs.astype(np.float32)

    def embed_sequence_mean(self, images: list[np.ndarray]) -> np.ndarray:
        """Mean embedding across all frames → (dim,)."""
        if not images:
            self._load()
            return np.zeros(self.dim, dtype=np.float32)
        return self.embed(images).mean(axis=0)

    @property
    def output_dim(self) -> int:
        self._load()
        return self.dim
