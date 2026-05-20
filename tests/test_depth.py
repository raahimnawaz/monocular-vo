"""Sanity check for the Depth Anything wrapper.

Gated behind `downloads_model` because it pulls model weights from HuggingFace.
Run locally with: pytest -m downloads_model
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.downloads_model
def test_depth_output_shape_and_units() -> None:
    from monocular_vo.depth import DepthAnything

    # A simple gradient image is enough to check shape + that values are positive.
    img = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1))
    img = np.stack([img, img, img], axis=-1)

    model = DepthAnything()
    depth = model.predict(img)

    assert depth.shape == (480, 640)
    assert depth.dtype == np.float32
    assert np.isfinite(depth).all()
    assert (depth > 0).all(), "metric depth should be strictly positive"
