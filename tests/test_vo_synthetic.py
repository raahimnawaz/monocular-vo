"""Synthetic VO sanity test (CI-safe — no camera, no HF model).

Generates a deterministic 3D point cloud, projects it through known camera
poses, hands the synthetic features + ground-truth depth to the VO step,
and asserts that the recovered motion matches the input.
"""
from __future__ import annotations

import cv2
import numpy as np

from monocular_vo.vo import Trajectory, backproject, estimate_relative_pose


def _make_intrinsics(width: int = 640, height: int = 480, fov_deg: float = 60.0) -> np.ndarray:
    f = (width / 2) / np.tan(np.deg2rad(fov_deg) / 2)
    K = np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K


def _project(K: np.ndarray, pts3d: np.ndarray) -> np.ndarray:
    proj = (K @ pts3d.T).T
    return proj[:, :2] / proj[:, 2:3]


def test_recovers_known_motion() -> None:
    rng = np.random.default_rng(0)
    K = _make_intrinsics()

    # 200 random points 2-8 m in front of the camera, +/- 1 m laterally
    n = 200
    pts3d_prev = np.stack(
        [
            rng.uniform(-1.0, 1.0, n),
            rng.uniform(-0.6, 0.6, n),
            rng.uniform(2.0, 8.0, n),
        ],
        axis=-1,
    )
    uv_prev = _project(K, pts3d_prev)

    # Apply a 30 cm forward step, no rotation
    t_true = np.array([0.0, 0.0, 0.3])
    R_true = np.eye(3)
    pts3d_curr = (R_true @ pts3d_prev.T).T + t_true
    uv_curr = _project(K, pts3d_curr)

    # Build a sparse "depth map" that returns ground-truth z at the prev pixels
    depth_map = np.full((480, 640), np.nan, dtype=np.float64)
    for (u, v), z in zip(uv_prev, pts3d_prev[:, 2], strict=False):
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < 640 and 0 <= vi < 480:
            depth_map[vi, ui] = z

    pts3d_recovered = backproject(uv_prev, depth_map, K)
    finite = np.isfinite(pts3d_recovered).all(axis=1).sum()
    assert finite >= 0.9 * n, f"only {finite}/{n} backprojections finite"

    result = estimate_relative_pose(pts3d_recovered, uv_curr, K)
    assert result.num_inliers > 100, f"too few inliers: {result.num_inliers}"

    assert np.allclose(result.t, t_true, atol=0.02), f"t={result.t}, expected {t_true}"
    rvec, _ = cv2.Rodrigues(result.R)
    assert np.linalg.norm(rvec) < 0.02, f"unexpected rotation: rvec={rvec.ravel()}"


def test_trajectory_accumulation() -> None:
    # PnP's t maps prev-frame points to curr-frame coords:
    #   x_curr = R @ x_prev + t
    # If the camera walks forward (along +z in world), world points end up at
    # smaller z in the current frame, so t.z is negative.
    traj = Trajectory()
    R = np.eye(3)
    t_forward = np.array([0.0, 0.0, -0.5])  # 0.5 m forward step
    for _ in range(4):
        traj.update(R, t_forward)
    assert np.allclose(traj.positions[-1], [0.0, 0.0, 2.0])
    assert np.isclose(traj.length_m, 2.0)


def test_recovers_rotation_only_motion() -> None:
    rng = np.random.default_rng(1)
    K = _make_intrinsics()

    n = 200
    pts3d_prev = np.stack(
        [
            rng.uniform(-1.0, 1.0, n),
            rng.uniform(-0.6, 0.6, n),
            rng.uniform(2.0, 6.0, n),
        ],
        axis=-1,
    )
    uv_prev = _project(K, pts3d_prev)

    angle = np.deg2rad(5.0)
    R_true = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    pts3d_curr = (R_true @ pts3d_prev.T).T
    uv_curr = _project(K, pts3d_curr)

    depth_map = np.full((480, 640), np.nan, dtype=np.float64)
    for (u, v), z in zip(uv_prev, pts3d_prev[:, 2], strict=False):
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < 640 and 0 <= vi < 480:
            depth_map[vi, ui] = z

    pts3d_recovered = backproject(uv_prev, depth_map, K)
    result = estimate_relative_pose(pts3d_recovered, uv_curr, K)
    assert result.num_inliers > 100

    rvec, _ = cv2.Rodrigues(result.R)
    recovered_angle = float(np.linalg.norm(rvec))
    assert abs(recovered_angle - angle) < np.deg2rad(0.5)
    assert np.linalg.norm(result.t) < 0.05
