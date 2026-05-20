"""End-to-end VO: video + intrinsics -> trajectory + metrics.

Pipeline per frame pair:
    1. ORB features in t-1 and t, match with ratio test
    2. Depth Anything v2 predicts metric depth on t-1
    3. Backproject matched keypoints in t-1 to 3D (meters)
    4. solvePnPRansac for relative pose, accumulate trajectory
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from monocular_vo.calibrate import Calibration
from monocular_vo.depth import DEFAULT_MODEL, DepthAnything
from monocular_vo.visualize import plot_trajectory
from monocular_vo.vo import Trajectory, step


def iter_frames(video_path: Path, stride: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--calibration", type=Path, default=Path("data/calibration.npz"))
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=2, help="process every Nth frame")
    ap.add_argument(
        "--ground-truth-length",
        type=float,
        default=None,
        help="optional tape-measured path length (meters) for scale-error metric",
    )
    ap.add_argument("--depth-model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"no such video: {args.video}")
    if not args.calibration.exists():
        raise SystemExit(
            f"no such calibration: {args.calibration}\nrun scripts/run_calibration.py first"
        )

    calib = Calibration.load(args.calibration)
    K = calib.K
    print(f"loaded calibration (reprojection error {calib.reprojection_error:.3f} px)")

    out_dir = args.output_dir or (args.video.parent / args.video.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading depth model: {args.depth_model}")
    depth_model = DepthAnything(model_id=args.depth_model)
    print(f"  device: {depth_model.device}")

    trajectory = Trajectory()
    runtimes_depth: list[float] = []
    runtimes_vo: list[float] = []
    inliers_log: list[int] = []
    skipped = 0

    prev_gray: np.ndarray | None = None
    prev_depth: np.ndarray | None = None

    frames = list(iter_frames(args.video, args.stride))
    print(f"processing {len(frames)} frames (stride={args.stride})")

    for _, frame in tqdm(frames, desc="VO"):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t0 = time.perf_counter()
        depth = depth_model.predict(frame)
        t1 = time.perf_counter()
        runtimes_depth.append((t1 - t0) * 1000.0)

        if prev_gray is not None and prev_depth is not None:
            t2 = time.perf_counter()
            result = step(prev_gray, gray, prev_depth, K)
            t3 = time.perf_counter()
            runtimes_vo.append((t3 - t2) * 1000.0)
            inliers_log.append(result.num_inliers)
            if result.degenerate:
                skipped += 1
            else:
                trajectory.update(result.R, result.t)

        prev_gray = gray
        prev_depth = depth

    positions = np.asarray(trajectory.positions)
    csv_path = out_dir / "trajectory.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "x", "y", "z"])
        for i, p in enumerate(positions):
            w.writerow([i, float(p[0]), float(p[1]), float(p[2])])

    png_path = out_dir / "trajectory.png"
    plot_trajectory(positions, png_path, ground_truth_length=args.ground_truth_length)

    metrics = {
        "frames_processed": len(frames),
        "steps_accumulated": len(positions) - 1,
        "steps_skipped_degenerate": skipped,
        "trajectory_length_m": trajectory.length_m,
        "endpoint_m": [float(x) for x in trajectory.endpoint_m],
        "endpoint_distance_m": float(np.linalg.norm(trajectory.endpoint_m)),
        "mean_depth_runtime_ms": float(np.mean(runtimes_depth)) if runtimes_depth else 0.0,
        "mean_vo_runtime_ms": float(np.mean(runtimes_vo)) if runtimes_vo else 0.0,
        "median_inliers": float(np.median(inliers_log)) if inliers_log else 0.0,
    }
    if args.ground_truth_length is not None and args.ground_truth_length > 0:
        scale_err = (
            (trajectory.length_m - args.ground_truth_length) / args.ground_truth_length
        )
        metrics["ground_truth_length_m"] = args.ground_truth_length
        metrics["scale_error_pct"] = float(scale_err * 100.0)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"wrote {csv_path}, {png_path}, {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
