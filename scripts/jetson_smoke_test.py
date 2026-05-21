"""Device-agnostic Depth Anything v2 smoke test.

Loads the depth model on the best available device (CUDA / MPS / CPU), runs
warm-up inferences, then times N more, and prints both latency stats and
sanity stats on the depth output.

This script is the Jetson PLAN.md Phase 1 deliverable. Running it on the Mac
(MPS) right now establishes a baseline; running the same script on the Jetson
when the board arrives is the verification step — only the device label
should change (`mps` -> `cuda`) and the latency should drop (~227 ms M5 MPS
-> ~60-80 ms Jetson Orin CUDA pre-TensorRT).

Writes a JSON summary to `data/smoke_<device>.json` so the Mac and Jetson
runs can be diffed mechanically later.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from monocular_vo.depth import DEFAULT_MODEL, DepthAnything


def _load_frame(video: Path, frame_idx: int) -> tuple[np.ndarray, str]:
    """Sample a frame from `video`, falling back to a deterministic synthetic
    pattern if the video isn't on disk (e.g., fresh checkout on the Jetson)."""
    if video.exists():
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if ok:
            return frame, f"{video.name} frame {frame_idx}"

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    return frame, "synthetic 640x480 random frame (deterministic, seed=0)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--video",
        type=Path,
        default=Path("data/sequences/hallway-5m.mp4"),
        help="video to sample a frame from; falls back to synthetic if missing",
    )
    ap.add_argument("--frame-idx", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=3, help="warm-up iters (not timed)")
    ap.add_argument("--iters", type=int, default=10, help="timed iters")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write the JSON summary (default: data/smoke_<device>.json)",
    )
    args = ap.parse_args()

    frame, source = _load_frame(args.video, args.frame_idx)

    print("=" * 60)
    print("Depth Anything v2 smoke test")
    print("=" * 60)
    print(f"  PyTorch:        {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  MPS available:  {torch.backends.mps.is_available()}")
    print(f"  Source:         {source}")
    print(f"  Frame shape:    {frame.shape}")
    print(f"  Model:          {args.model}")
    print()

    print("Loading model...")
    t0 = time.perf_counter()
    model = DepthAnything(model_id=args.model)
    load_time_s = time.perf_counter() - t0
    print(f"  Device: {model.device}")
    print(f"  Load time: {load_time_s:.2f} s")
    print()

    print(f"Warm-up ({args.warmup} iters, not timed)...")
    for _ in range(args.warmup):
        _ = model.predict(frame)

    print(f"Timing ({args.iters} iters)...")
    times_ms: list[float] = []
    depth: np.ndarray | None = None
    for _ in range(args.iters):
        t0 = time.perf_counter()
        depth = model.predict(frame)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    assert depth is not None
    times = np.array(times_ms)
    finite = depth[np.isfinite(depth)]

    print()
    print("Latency (ms, after warm-up):")
    print(f"  mean   {times.mean():7.1f}")
    print(f"  median {np.median(times):7.1f}")
    print(f"  min    {times.min():7.1f}")
    print(f"  max    {times.max():7.1f}")
    print(f"  stddev {times.std():7.1f}")
    print()
    print("Depth sanity:")
    print(f"  output shape:        {depth.shape}")
    print(f"  output dtype:        {depth.dtype}")
    print(f"  min/median/max (m):  {finite.min():.3f} / {np.median(finite):.3f} / {finite.max():.3f}")
    print(f"  finite pixels:       {finite.size / depth.size * 100:.1f}%")

    device_tag = str(model.device).replace(":", "_")
    out_path = args.out or Path(f"data/smoke_{device_tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "device": str(model.device),
        "model": args.model,
        "torch_version": torch.__version__,
        "source": source,
        "frame_shape": list(frame.shape),
        "load_time_s": load_time_s,
        "warmup_iters": args.warmup,
        "timed_iters": args.iters,
        "latency_ms": {
            "mean": float(times.mean()),
            "median": float(np.median(times)),
            "min": float(times.min()),
            "max": float(times.max()),
            "stddev": float(times.std()),
        },
        "depth_meters": {
            "min": float(finite.min()),
            "median": float(np.median(finite)),
            "max": float(finite.max()),
            "finite_fraction": float(finite.size / depth.size),
        },
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"Wrote summary -> {out_path}")


if __name__ == "__main__":
    main()
