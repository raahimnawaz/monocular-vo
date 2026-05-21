"""Tests for the pose-graph back-end.

Builds a synthetic square-loop trajectory with intentionally-noisy odometry,
then verifies that adding a single loop-closure factor dramatically reduces
the endpoint drift after optimization.
"""
from __future__ import annotations

import gtsam
import numpy as np

from monocular_vo.slam import PoseGraph, pnp_to_relative_pose


def _euler_z(deg: float) -> np.ndarray:
    """Rotation about z-axis by `deg` degrees."""
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _relative_pose_to_pnp(R_rel: np.ndarray, t_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of pnp_to_relative_pose for test construction.

    Given a gtsam-style relative pose (camera-to-world convention), produce
    the (R_pnp, t_pnp) that the VO front-end would emit for that motion.
    """
    R_pnp = R_rel.T
    t_pnp = -R_pnp @ t_rel
    return R_pnp, t_pnp


def test_pnp_round_trip() -> None:
    rng = np.random.default_rng(0)
    R = _euler_z(15.0)
    t = rng.normal(scale=0.5, size=3)
    rel = pnp_to_relative_pose(R, t)
    # Apply rel to identity pose; should map a "kf" point correctly into "cur" frame
    p_kf = np.array([1.0, 2.0, 3.0])
    p_cur_expected = R @ p_kf + t
    # The inverse-pose relationship: rel is Pose_kf.between(Pose_cur), so going
    # from cur's world coords back into kf's frame:
    pose_kf = gtsam.Pose3()
    pose_cur = pose_kf.compose(rel)
    p_world = pose_kf.transformFrom(p_kf)  # kf is identity, so p_world == p_kf
    p_cur = pose_cur.transformTo(p_world)
    assert np.allclose(p_cur, p_cur_expected, atol=1e-9)


def test_square_loop_closure_reduces_drift() -> None:
    """4-pose square loop, each step = forward 4 m then turn left 90 deg.
    Odometry has small bias; loop closure should cancel the accumulated drift."""
    side = 4.0
    # Each step in pose_i's frame: translate forward 4m, then rotate left 90 deg.
    # Pose_{i+1} = Pose_i * step, so this composition naturally closes after 4 steps.
    step_R = _euler_z(90.0)
    step_t = np.array([side, 0.0, 0.0])
    motions = [(step_R, step_t.copy()) for _ in range(4)]

    # Ground-truth integration sanity check
    R_gt: list[np.ndarray] = [np.eye(3)]
    t_gt: list[np.ndarray] = [np.zeros(3)]
    for R_step, t_step in motions:
        t_gt.append(t_gt[-1] + R_gt[-1] @ t_step)
        R_gt.append(R_gt[-1] @ R_step)
    assert np.linalg.norm(t_gt[-1]) < 1e-9, f"GT loop not closed: {t_gt[-1]}"

    # Add a small rotational bias to each odometry measurement (drift source)
    biased_motions = []
    for R_step, t_step in motions:
        R_biased = _euler_z(3.0) @ R_step
        biased_motions.append((R_biased, t_step.copy()))

    # Build graph using the biased odometry
    graph = PoseGraph()
    # Initial estimate is the (biased) integrated trajectory
    R_world: list[np.ndarray] = [np.eye(3)]
    t_world: list[np.ndarray] = [np.zeros(3)]
    graph.add_prior(0, R_world[0], t_world[0])
    for i, (R_step, t_step) in enumerate(biased_motions):
        R_new = R_world[-1] @ R_step
        t_new = t_world[-1] + R_world[-1] @ t_step
        R_world.append(R_new)
        t_world.append(t_new)
        graph.add_initial_estimate(i + 1, R_new, t_new)
        # The "PnP" inputs for this odometry step
        R_pnp, t_pnp = _relative_pose_to_pnp(R_step, t_step)
        graph.add_odometry(i, i + 1, R_pnp, t_pnp)

    # Drift before optimization (no loop closure yet)
    drift_before = float(np.linalg.norm(t_world[-1]))
    assert drift_before > 0.3, f"setup error: expected nontrivial drift, got {drift_before:.3f}"

    # Add a loop closure: keyframe 4 (last) revisits keyframe 0 (start) with
    # the true relative pose ~ identity (we know they're the same place).
    R_loop = np.eye(3)
    t_loop = np.zeros(3)
    R_loop_pnp, t_loop_pnp = _relative_pose_to_pnp(R_loop, t_loop)
    graph.add_loop_closure(0, 4, R_loop_pnp, t_loop_pnp, inlier_weight=5.0)

    optimized = graph.optimize()
    drift_after = float(np.linalg.norm(optimized[4][1]))

    assert drift_after < 0.05, f"optimization didn't tighten the loop: {drift_after:.3f} m drift"
    print(f"drift {drift_before:.3f} m -> {drift_after:.3f} m after loop closure")
