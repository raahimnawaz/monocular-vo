"""Run monocular VO + SLAM on a TUM RGB-D sequence.

Reports ATE/RPE against the mocap ground truth, before and after pose-graph
optimization with loop closures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from monocular_vo import tum
from monocular_vo.depth import DEFAULT_MODEL, DepthAnything
from monocular_vo.eval import ate_rmse, rpe_translation, scale_error
from monocular_vo.keyframe import Keyframe, KeyframeSelector
from monocular_vo.loop import LoopDetector
from monocular_vo.slam import PoseGraph, relative_pnp
from monocular_vo.visualize import plot_trajectory
from monocular_vo.vo import Trajectory, detect_features, step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequence", type=Path, required=True, help="TUM sequence root dir")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--depth-model", default=DEFAULT_MODEL)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--keyframe-min-translation-m", type=float, default=0.10)
    ap.add_argument("--keyframe-min-rotation-deg", type=float, default=8.0)
    ap.add_argument("--loop-temporal-skip", type=int, default=15)
    ap.add_argument("--loop-match-threshold", type=int, default=80)
    ap.add_argument("--loop-verify-inliers", type=int, default=60)
    ap.add_argument("--loop-max-translation-m", type=float, default=1.5)
    args = ap.parse_args()

    seq = tum.load(args.sequence)
    print(f"loaded TUM {args.sequence.name}: {len(seq)} frames")
    print(f"  GT matched: {int(seq.gt_matched_mask.sum())} / {len(seq)}")

    depth_model = DepthAnything(model_id=args.depth_model)
    print(f"  depth on {depth_model.device}")

    out_dir = args.output_dir or (args.sequence / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(seq) if args.max_frames is None else min(len(seq), args.max_frames)

    # --- VO front-end -----------------------------------------------------
    trajectory = Trajectory()
    prev_gray: np.ndarray | None = None
    prev_depth: np.ndarray | None = None
    per_frame_depths: list[np.ndarray] = []  # cached for keyframe creation later
    per_frame_grays: list[np.ndarray] = []
    sampled_indices: list[int] = []
    inliers_log: list[int] = []
    skipped = 0

    for i in tqdm(range(0, n, args.stride), desc="VO"):
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
                sampled_indices.append(i)
        else:
            sampled_indices.append(i)

        per_frame_grays.append(gray)
        per_frame_depths.append(depth)
        prev_gray = gray
        prev_depth = depth

    vo_positions = np.asarray(trajectory.positions)
    np.save(out_dir / "vo_positions.npy", vo_positions)
    plot_trajectory(vo_positions, out_dir / "vo_trajectory.png")

    # --- Keyframes + back-end ---------------------------------------------
    selector = KeyframeSelector(
        min_translation_m=args.keyframe_min_translation_m,
        min_rotation_deg=args.keyframe_min_rotation_deg,
    )
    keyframes: list[Keyframe] = []
    pose_graph = PoseGraph()
    loop_detector = LoopDetector(
        K=seq.K,
        candidate_match_threshold=args.loop_match_threshold,
        temporal_skip=args.loop_temporal_skip,
        verification_inlier_threshold=args.loop_verify_inliers,
        max_relative_translation_m=(
            args.loop_max_translation_m if args.loop_max_translation_m > 0 else None
        ),
    )
    loop_closures: list[tuple[int, int, int]] = []

    next_kf_id = 0
    for j, (idx_in_video, R_now, t_now) in enumerate(
        zip(sampled_indices, trajectory.rotations, trajectory.positions, strict=False)
    ):
        local_idx = idx_in_video // args.stride
        gray = per_frame_grays[local_idx] if local_idx < len(per_frame_grays) else None
        if gray is None:
            continue
        last_inliers = inliers_log[j - 1] if j > 0 and j - 1 < len(inliers_log) else 0
        if not selector.should_insert(R_now, t_now, last_inliers):
            continue
        feats = detect_features(gray)
        kp_uv = np.array([kp.pt for kp in feats.keypoints], dtype=np.float64)
        depth = per_frame_depths[local_idx]
        kf = Keyframe(
            index=idx_in_video,
            keyframe_id=next_kf_id,
            R_world=R_now.copy(),
            t_world=t_now.copy(),
            descriptors=feats.descriptors,
            keypoints_uv=kp_uv,
            depth=depth.copy(),
        )
        if next_kf_id == 0:
            pose_graph.add_prior(0, kf.R_world, kf.t_world)
        else:
            prev_kf = keyframes[-1]
            pose_graph.add_initial_estimate(kf.keyframe_id, kf.R_world, kf.t_world)
            R_pnp, t_pnp = relative_pnp(prev_kf.R_world, prev_kf.t_world, kf.R_world, kf.t_world)
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
                loop_closures.append((closure.to_id, closure.from_id, closure.num_inliers))
        keyframes.append(kf)
        selector.insert(kf)
        next_kf_id += 1

    print(f"  {len(keyframes)} keyframes, {len(loop_closures)} loop closures")

    if len(keyframes) >= 2:
        optimized = pose_graph.optimize()
        slam_positions = np.array([optimized[kf.keyframe_id][1] for kf in keyframes])
    else:
        slam_positions = np.array([kf.t_world for kf in keyframes])
    np.save(out_dir / "slam_positions.npy", slam_positions)
    plot_trajectory(slam_positions, out_dir / "slam_trajectory.png")

    # Keyframe ground-truth positions (one GT pose per keyframe)
    kf_gt = np.array([seq.gt_positions[kf.index] for kf in keyframes])
    kf_gt_mask = np.array([bool(seq.gt_matched_mask[kf.index]) for kf in keyframes])

    metrics: dict = {
        "sequence": seq.name,
        "frames_processed": len(sampled_indices),
        "keyframes": len(keyframes),
        "loop_closures": len(loop_closures),
        "loop_closure_pairs": loop_closures,
        "steps_skipped_degenerate": skipped,
        "median_inliers": float(np.median(inliers_log)) if inliers_log else 0.0,
        "gt_matched_keyframes": int(kf_gt_mask.sum()),
    }

    if kf_gt_mask.sum() >= 5:
        gt_kf = kf_gt[kf_gt_mask]
        vo_kf = np.array([kf.t_world for kf in keyframes])[kf_gt_mask]
        slam_kf = slam_positions[kf_gt_mask]
        metrics["vo_ate_rmse_m"] = ate_rmse(vo_kf, gt_kf, with_scale=True)
        metrics["slam_ate_rmse_m"] = ate_rmse(slam_kf, gt_kf, with_scale=True)
        metrics["vo_rpe_m"] = rpe_translation(vo_kf, gt_kf, delta=1, with_scale=True)
        metrics["slam_rpe_m"] = rpe_translation(slam_kf, gt_kf, delta=1, with_scale=True)
        metrics["vo_scale_error_pct"] = scale_error(vo_kf, gt_kf)
        metrics["slam_scale_error_pct"] = scale_error(slam_kf, gt_kf)
        metrics["ate_reduction_pct"] = (
            100.0 * (metrics["vo_ate_rmse_m"] - metrics["slam_ate_rmse_m"]) / metrics["vo_ate_rmse_m"]
        )
        # Save matched ground-truth keyframe trajectory for plotting
        np.save(out_dir / "gt_keyframe_positions.npy", gt_kf)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
