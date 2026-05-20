"""Depth Anything v2 (Metric) inference wrapper.

Wraps HuggingFace `transformers` Depth Anything v2 Metric models. The metric
checkpoints (`*-Metric-Indoor-*` and `*-Metric-Outdoor-*`) output absolute
depth in meters, which is what makes the downstream PnP step recover a
trajectory in real-world units.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class DepthAnything:
    model_id: str = DEFAULT_MODEL
    device: torch.device | None = None
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.device = self.device or _select_device()
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Predict per-pixel depth in meters from a BGR (OpenCV) image.

        Returns a float32 `(H, W)` array.
        """
        image_rgb = image_bgr[..., ::-1]
        pil = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # Resize back to the input resolution.
        predicted = outputs.predicted_depth  # (1, h', w')
        h, w = image_bgr.shape[:2]
        depth = torch.nn.functional.interpolate(
            predicted.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze()
        return depth.detach().cpu().float().numpy()
