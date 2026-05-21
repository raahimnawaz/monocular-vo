"""Pose-graph back-end (gtsam Levenberg-Marquardt).

Conventions
-----------
Poses are stored as `gtsam.Pose3`, which represents a *camera-to-world*
transform: ``world_X = Pose * camera_X``.

This matches `Trajectory` in `vo.py`, where ``rotations[i]`` is the rotation
matrix from camera i's frame to the world frame and ``positions[i]`` is the
camera origin expressed in world coords. (The variable naming there is a
historical artefact — the math is correct, see `test_vo_synthetic`.)

PnP-to-BetweenFactor conversion
-------------------------------
The VO front-end's `solvePnPRansac` gives ``(R_pnp, t_pnp)`` such that

    x_cur = R_pnp @ x_kf + t_pnp           (1)

i.e. it maps a 3D point from the older camera's frame into the newer
camera's frame. A gtsam `BetweenFactorPose3(kf, cur, T, noise)` instead wants
``T = Pose_kf.between(Pose_cur)`` — the relative pose from kf to cur in the
camera-to-world convention. Algebra gives

    T.rotation    =  R_pnp.T
    T.translation = -R_pnp.T @ t_pnp

i.e. ``T`` is the SE(3) inverse of ``(R_pnp, t_pnp)``. `pnp_to_relative_pose`
encapsulates this conversion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gtsam
import numpy as np


def pnp_to_relative_pose(R_pnp: np.ndarray, t_pnp: np.ndarray) -> gtsam.Pose3:
    """Convert a PnP (R, t) pair to the gtsam relative-pose convention."""
    R_inv = R_pnp.T
    t_inv = -R_inv @ t_pnp
    return gtsam.Pose3(gtsam.Rot3(R_inv), t_inv)


def matrix_to_pose3(R: np.ndarray, t: np.ndarray) -> gtsam.Pose3:
    return gtsam.Pose3(gtsam.Rot3(R), t)


def pose3_to_matrix(pose: gtsam.Pose3) -> tuple[np.ndarray, np.ndarray]:
    return pose.rotation().matrix(), pose.translation()


def relative_pnp(
    R_old: np.ndarray,
    t_old: np.ndarray,
    R_new: np.ndarray,
    t_new: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the PnP-convention (R, t) that maps points in the old camera
    frame into the new camera frame, given both poses in camera-to-world form."""
    R_pnp = R_new.T @ R_old
    t_pnp = R_new.T @ (t_old - t_new)
    return R_pnp, t_pnp


@dataclass
class PoseGraphConfig:
    odom_sigma_rot_deg: float = 1.0
    odom_sigma_trans_m: float = 0.05
    loop_sigma_rot_deg: float = 2.0
    loop_sigma_trans_m: float = 0.10
    prior_sigma_rot_deg: float = 0.01
    prior_sigma_trans_m: float = 0.001


def _diag_noise(sigma_rot_deg: float, sigma_trans_m: float) -> gtsam.noiseModel.Diagonal:
    sigma_rot = np.deg2rad(sigma_rot_deg)
    sigmas = np.array(
        [sigma_rot, sigma_rot, sigma_rot, sigma_trans_m, sigma_trans_m, sigma_trans_m]
    )
    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)


@dataclass
class PoseGraph:
    config: PoseGraphConfig = field(default_factory=PoseGraphConfig)
    graph: gtsam.NonlinearFactorGraph = field(default_factory=gtsam.NonlinearFactorGraph)
    initial: gtsam.Values = field(default_factory=gtsam.Values)
    inserted: set[int] = field(default_factory=set)

    def add_prior(self, pose_id: int, R: np.ndarray, t: np.ndarray) -> None:
        noise = _diag_noise(self.config.prior_sigma_rot_deg, self.config.prior_sigma_trans_m)
        self.graph.add(gtsam.PriorFactorPose3(pose_id, matrix_to_pose3(R, t), noise))
        if pose_id not in self.inserted:
            self.initial.insert(pose_id, matrix_to_pose3(R, t))
            self.inserted.add(pose_id)

    def add_initial_estimate(self, pose_id: int, R: np.ndarray, t: np.ndarray) -> None:
        if pose_id not in self.inserted:
            self.initial.insert(pose_id, matrix_to_pose3(R, t))
            self.inserted.add(pose_id)

    def add_odometry(
        self,
        from_id: int,
        to_id: int,
        R_pnp: np.ndarray,
        t_pnp: np.ndarray,
    ) -> None:
        noise = _diag_noise(self.config.odom_sigma_rot_deg, self.config.odom_sigma_trans_m)
        rel = pnp_to_relative_pose(R_pnp, t_pnp)
        self.graph.add(gtsam.BetweenFactorPose3(from_id, to_id, rel, noise))

    def add_loop_closure(
        self,
        from_id: int,
        to_id: int,
        R_pnp: np.ndarray,
        t_pnp: np.ndarray,
        inlier_weight: float = 1.0,
    ) -> None:
        """Add a loop-closure between an *older* (from) and *newer* (current) keyframe.

        `from_id` is the older keyframe, `to_id` is the current keyframe.
        `(R_pnp, t_pnp)` is the PnP result that maps `from`'s 3D points into
        `to`'s camera frame (matching the VO step convention).

        `inlier_weight` lets the caller upweight closures with many inliers
        (larger weight → smaller sigma, i.e. trust the closure more).
        """
        sigma_scale = max(1.0 / inlier_weight, 0.5)
        noise = _diag_noise(
            self.config.loop_sigma_rot_deg * sigma_scale,
            self.config.loop_sigma_trans_m * sigma_scale,
        )
        rel = pnp_to_relative_pose(R_pnp, t_pnp)
        self.graph.add(gtsam.BetweenFactorPose3(from_id, to_id, rel, noise))

    def optimize(self, max_iter: int = 50) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(max_iter)
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, params)
        result = optimizer.optimize()
        out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for key in self.inserted:
            pose = result.atPose3(key)
            out[key] = pose3_to_matrix(pose)
        return out
