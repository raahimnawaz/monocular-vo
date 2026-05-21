"""End-to-end mini SLAM: VO front-end + keyframes + loop closure + pose-graph.

Pipeline:
    1. Iterate frames, run the VO front-end (depth-PnP) to get per-frame motion.
    2. Promote some frames to keyframes using KeyframeSelector heuristics.
    3. Between consecutive keyframes, add an odometry factor (computed from the
       VO trajectory; this is the front-end's incremental estimate).
    4. For each new keyframe, search for loop-closure candidates among prior
       keyframes (descriptor match + PnP geometric verification).
    5. After the run, batch-optimize the full pose graph with Levenberg-Marquardt.
    6. Save both the raw VO-only and the optimized SLAM trajectories so the
       ablation (with/without loop closure) is reproducible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from monocular_vo.calibrate import Calibration
from monocular_vo.depth import DEFAULT_MODEL, DepthAnything
from monocular_vo.keyframe import Keyframe, KeyframeSelector
from monocular_vo.loop import LoopDetector
from monocular_vo.slam import PoseGraph, relative_pnp
from monocular_vo.visualize import plot_trajectory
from monocular_vo.vo import Trajectory, detect_features, step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--calibration", type=Path, default=Path("data/calibration.npz"))
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--depth-model", default=DEFAULT_MODEL)
    ap.add_argument("--keyframe-min-translation-m", type=float, default=0.30)
    ap.add_argument("--keyframe-min-rotation-deg", type=float, default=10.0)
    # Loop closure defaults are conservative — false positives in repetitive
    # scenes (hallways, identical-looking rooms) are worse than missed closures.
    # For real-loop sequences lower temporal-skip and verify-inliers.
    ap.add_argument("--loop-temporal-skip", type=int, default=20)
    ap.add_argument("--loop-match-threshold", type=int, default=100)
    ap.add_argument("--loop-verify-inliers", type=int, default=150)
    ap.add_argument(
        "--loop-max-translation-m",
        type=float,
        default=3.0,
        help="reject loop candidates further than this many meters apart "
        "(use 0 to disable; set higher for outdoor/long-range sequences)",
    )
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"no such video: {args.video}")
    if not args.calibration.exists():
        raise SystemExit(f"no such calibration: {args.calibration}")

    calib = Calibration.load(args.calibration)
    K = calib.K
    print(f"calibration loaded (reproj err {calib.reprojection_error:.3f} px)")

    depth_model = DepthAnything(model_id=args.depth_model)
    print(f"depth model: {args.depth_model} on {depth_model.device}")

    out_dir = args.output_dir or (args.video.parent / (args.video.stem + "_slam"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # VO state
    trajectory = Trajectory()
    prev_gray: np.ndarray | None = None
    prev_depth: np.ndarray | None = None

    # Keyframe / SLAM state
    selector = KeyframeSelector(
        min_translation_m=args.keyframe_min_translation_m,
        min_rotation_deg=args.keyframe_min_rotation_deg,
    )
    keyframes: list[Keyframe] = []
    pose_graph = PoseGraph()
    loop_detector = LoopDetector(
        K=K,
        candidate_match_threshold=args.loop_match_threshold,
        temporal_skip=args.loop_temporal_skip,
        verification_inlier_threshold=args.loop_verify_inliers,
        max_relative_translation_m=(
            args.loop_max_translation_m if args.loop_max_translation_m > 0 else None
        ),
    )
    loop_closures_added: list[tuple[int, int, int]] = []  # (from, to, inliers)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")

    pbar = tqdm(desc="slam")
    frame_idx = 0
    next_kf_id = 0
    last_inliers = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            depth = depth_model.predict(frame)

            if prev_gray is not None and prev_depth is not None:
                result = step(prev_gray, gray, prev_depth, K)
                last_inliers = result.num_inliers
                if not result.degenerate:
                    trajectory.update(result.R, result.t)

            # Decide whether to promote this frame to a keyframe.
            R_now = trajectory.rotations[-1]
            t_now = trajectory.positions[-1]
            if selector.should_insert(R_now, t_now, last_inliers):
                feats = detect_features(gray)
                kp_uv = np.array([kp.pt for kp in feats.keypoints], dtype=np.float64)
                kf = Keyframe(
                    index=frame_idx,
                    keyframe_id=next_kf_id,
                    R_world=R_now.copy(),
                    t_world=t_now.copy(),
                    descriptors=feats.descriptors,
                    keypoints_uv=kp_uv,
                    depth=depth.copy(),
                )
                # Add this keyframe to the pose graph
                if next_kf_id == 0:
                    pose_graph.add_prior(0, kf.R_world, kf.t_world)
                else:
                    prev_kf = keyframes[-1]
                    pose_graph.add_initial_estimate(kf.keyframe_id, kf.R_world, kf.t_world)
                    R_pnp, t_pnp = relative_pnp(
                        prev_kf.R_world, prev_kf.t_world, kf.R_world, kf.t_world
                    )
                    pose_graph.add_odometry(prev_kf.keyframe_id, kf.keyframe_id, R_pnp, t_pnp)

                    closure = loop_detector.detect(kf, keyframes)
                    if closure is not None:
                        inlier_weight = max(closure.num_inliers / 200.0, 1.0)
                        pose_graph.add_loop_closure(
                            closure.to_id,
                            closure.from_id,
                            closure.R,
                            closure.t,
                            inlier_weight=inlier_weight,
                        )
                        loop_closures_added.append(
                            (closure.to_id, closure.from_id, closure.num_inliers)
                        )

                keyframes.append(kf)
                selector.insert(kf)
                next_kf_id += 1

            prev_gray = gray
            prev_depth = depth
            frame_idx += 1
            pbar.update(1)
    finally:
        cap.release()
        pbar.close()

    print(f"{len(keyframes)} keyframes, {len(loop_closures_added)} loop closures detected")

    # Save VO-only trajectory (pre-optimization keyframe poses)
    vo_only_positions = np.array([kf.t_world for kf in keyframes])
    np.save(out_dir / "vo_only_positions.npy", vo_only_positions)

    if len(keyframes) < 2:
        print("not enough keyframes to optimize")
        return

    optimized = pose_graph.optimize()
    slam_positions = np.array([optimized[kf.keyframe_id][1] for kf in keyframes])
    np.save(out_dir / "slam_positions.npy", slam_positions)

    # Plots
    plot_trajectory(vo_only_positions, out_dir / "vo_only.png")
    plot_trajectory(slam_positions, out_dir / "slam.png")

    metrics = {
        "keyframes": len(keyframes),
        "loop_closures": len(loop_closures_added),
        "loop_closure_pairs": loop_closures_added,
        "vo_only_endpoint_m": [float(x) for x in vo_only_positions[-1]],
        "slam_endpoint_m": [float(x) for x in slam_positions[-1]],
        "vo_only_endpoint_distance_m": float(np.linalg.norm(vo_only_positions[-1])),
        "slam_endpoint_distance_m": float(np.linalg.norm(slam_positions[-1])),
        "vo_only_length_m": float(np.linalg.norm(np.diff(vo_only_positions, axis=0), axis=1).sum()),
        "slam_length_m": float(np.linalg.norm(np.diff(slam_positions, axis=0), axis=1).sum()),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
