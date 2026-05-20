"""Run monocular VO on a KITTI Odometry sequence and report ATE/RPE.

Expects KITTI Odometry layout under --root:
    <root>/sequences/<seq>/image_2/*.png
    <root>/sequences/<seq>/calib.txt
    <root>/poses/<seq>.txt   (sequences 00-10 have public ground truth)

Uses the outdoor metric variant of Depth Anything v2 by default since KITTI
is outdoor street footage; the indoor variant gives systematically wrong scale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from monocular_vo import kitti
from monocular_vo.depth import DepthAnything
from monocular_vo.eval import ate_rmse, rpe_translation, scale_error
from monocular_vo.visualize import plot_trajectory
from monocular_vo.vo import Trajectory, step

KITTI_OUTDOOR_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="KITTI odometry root dir")
    ap.add_argument("--seq", default="00")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--depth-model", default=KITTI_OUTDOOR_MODEL)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    seq = kitti.load(args.root, args.seq)
    print(f"loaded KITTI seq {args.seq}: {len(seq)} frames, K[0,0]={seq.K[0, 0]:.2f}")

    depth_model = DepthAnything(model_id=args.depth_model)
    print(f"depth model: {args.depth_model} on {depth_model.device}")

    trajectory = Trajectory()
    prev_gray: np.ndarray | None = None
    prev_depth: np.ndarray | None = None
    inliers_log: list[int] = []
    skipped = 0
    indices: list[int] = []

    n = len(seq) if args.max_frames is None else min(len(seq), args.max_frames)
    out_dir = args.output_dir or (args.root / "outputs" / args.seq)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(0, n, args.stride), desc=f"KITTI {args.seq}"):
        frame = seq.load_frame(i)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        depth = depth_model.predict(frame)

        if prev_gray is not None and prev_depth is not None:
            result = step(prev_gray, gray, prev_depth, seq.K)
            inliers_log.append(result.num_inliers)
            if result.degenerate:
                skipped += 1
            else:
                trajectory.update(result.R, result.t)
                indices.append(i)
        else:
            indices.append(i)

        prev_gray = gray
        prev_depth = depth

    est_positions = np.asarray(trajectory.positions)
    gt_positions = seq.gt_positions[indices[: len(est_positions)]]

    metrics: dict = {
        "sequence": args.seq,
        "frames_processed": n // args.stride,
        "steps_accumulated": len(est_positions) - 1,
        "steps_skipped_degenerate": skipped,
        "median_inliers": float(np.median(inliers_log)) if inliers_log else 0.0,
    }

    if gt_positions.any():
        metrics.update(
            {
                "ate_rmse_m": ate_rmse(est_positions, gt_positions, with_scale=True),
                "ate_rmse_no_scale_m": ate_rmse(est_positions, gt_positions, with_scale=False),
                "rpe_translation_m": rpe_translation(
                    est_positions, gt_positions, delta=1, with_scale=True
                ),
                "scale_error_pct": scale_error(est_positions, gt_positions),
                "gt_length_m": float(
                    np.linalg.norm(np.diff(gt_positions, axis=0), axis=1).sum()
                ),
                "est_length_m": float(
                    np.linalg.norm(np.diff(est_positions, axis=0), axis=1).sum()
                ),
            }
        )

    np.savez(out_dir / "trajectories.npz", est=est_positions, gt=gt_positions)
    plot_trajectory(est_positions, out_dir / "trajectory_est.png")
    if gt_positions.any():
        plot_trajectory(gt_positions, out_dir / "trajectory_gt.png")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
