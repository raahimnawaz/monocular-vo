"""Export Depth Anything v2 (Metric) to ONNX, then verify ONNX output against
PyTorch.

This is Phase 2a of `docs/JETSON_PLAN.md`. The output .onnx is the input to
Phase 2b on the Jetson (`trtexec --onnx=... --fp16 --saveEngine=depth.trt`).

Wraps the HF DPT model in a self-contained `nn.Module` that:
    1. Accepts raw RGB images in [0, 1] at a fixed shape, channels-first
    2. Applies the standard ImageNet normalization inline (so the ONNX
       graph is preprocessing-free at inference time — no Python
       preprocessor needed in C++ on the Jetson)
    3. Returns the predicted_depth tensor directly (no HF output object)

Fixed input shape (default 644x476) is divisible by 14 — DPT's patch size —
which avoids dynamic-shape complications when TensorRT compiles the engine.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from transformers import AutoModelForDepthEstimation

from monocular_vo.depth import DEFAULT_MODEL


class DepthAnythingExportable(nn.Module):
    """Wrapper that bakes ImageNet normalization into the graph and exposes
    a clean (image -> depth) forward signature for ONNX export."""

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.base = hf_model
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B, 3, H, W) float32 in [0, 1] -> depth: (B, H', W') in meters."""
        x = (image - self.mean) / self.std
        out = self.base(pixel_values=x)
        return out.predicted_depth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--height", type=int, default=476, help="must be divisible by 14")
    ap.add_argument("--width", type=int, default=644, help="must be divisible by 14")
    ap.add_argument("--output", type=Path, default=Path("artifacts/depth_anything_v2_small.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--atol", type=float, default=5e-3, help="ONNX vs PyTorch tolerance")
    args = ap.parse_args()

    if args.height % 14 != 0 or args.width % 14 != 0:
        raise SystemExit(
            f"DPT requires input dims divisible by 14; got {args.height}x{args.width}"
        )

    print("=" * 60)
    print("Depth Anything v2 ONNX export")
    print("=" * 60)
    print(f"  Model:  {args.model}")
    print(f"  Shape:  {args.height}x{args.width} (HxW)")
    print(f"  Opset:  {args.opset}")
    print(f"  Output: {args.output}")
    print()

    print("Loading PyTorch model...")
    hf_model = AutoModelForDepthEstimation.from_pretrained(args.model)
    hf_model.eval()
    wrapper = DepthAnythingExportable(hf_model)
    wrapper.eval()

    print(f"Building dummy input (1, 3, {args.height}, {args.width})...")
    torch.manual_seed(0)
    dummy = torch.rand(1, 3, args.height, args.width, dtype=torch.float32)

    print("Running PyTorch reference inference...")
    with torch.inference_mode():
        torch_out = wrapper(dummy)
    print(f"  PyTorch output shape: {tuple(torch_out.shape)}")
    print(f"  PyTorch output range: {torch_out.min():.3f} to {torch_out.max():.3f}")
    print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting ONNX (opset {args.opset})...")
    t0 = time.perf_counter()
    # Legacy TorchScript exporter (`dynamo=False`) — more stable than the new
    # dynamo path for transformer-style models on torch 2.12 + onnx-ir 0.1.x.
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(args.output),
        opset_version=args.opset,
        input_names=["image"],
        output_names=["depth"],
        dynamic_axes=None,  # static shapes -> simpler TRT compile
        do_constant_folding=True,
        dynamo=False,
    )
    export_s = time.perf_counter() - t0
    onnx_mb = args.output.stat().st_size / 1024 / 1024
    print(f"  Wrote {args.output} ({onnx_mb:.1f} MB) in {export_s:.1f}s")
    print()

    print("Verifying ONNX vs PyTorch with onnxruntime...")
    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    inputs = {sess.get_inputs()[0].name: dummy.numpy()}
    onnx_out = sess.run(None, inputs)[0]

    diff = np.abs(onnx_out - torch_out.numpy())
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    print(f"  ONNX output shape: {onnx_out.shape}")
    print(f"  ONNX output range: {onnx_out.min():.3f} to {onnx_out.max():.3f}")
    print(f"  max |ONNX - PyTorch|:  {max_abs:.6f} m")
    print(f"  mean |ONNX - PyTorch|: {mean_abs:.6f} m")
    print()

    if max_abs > args.atol:
        print(f"!!! Parity check FAILED: max abs diff {max_abs:.6f} > atol {args.atol}")
        print("    Likely causes: opset mismatch, unsupported op, dtype drift.")
        raise SystemExit(1)

    print(f"OK parity check passed (max diff {max_abs:.6f} <= atol {args.atol})")
    print()
    print("Next step (on Jetson):")
    print(f"  trtexec --onnx={args.output.name} --fp16 \\")
    print("          --saveEngine=depth.trt --workspace=4096")


if __name__ == "__main__":
    main()
