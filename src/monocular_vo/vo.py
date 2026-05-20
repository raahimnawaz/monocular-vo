"""Core VO pipeline: ORB features -> match -> depth backprojection -> PnP."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class FrameFeatures:
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray


@dataclass
class StepResult:
    R: np.ndarray  # 3x3, rotation from previous camera frame to current
    t: np.ndarray  # 3, translation in meters, expressed in previous frame
    num_inliers: int
    num_matches: int

    @property
    def degenerate(self) -> bool:
        return self.num_inliers < 30


@dataclass
class Trajectory:
    """Accumulated camera poses in a fixed world frame.

    `positions[i]` is the camera origin at step i. `rotations[i]` is the
    rotation from world to that camera frame.
    """

    positions: list[np.ndarray] = field(default_factory=lambda: [np.zeros(3)])
    rotations: list[np.ndarray] = field(default_factory=lambda: [np.eye(3)])

    def update(self, R_prev_to_curr: np.ndarray, t_in_prev: np.ndarray) -> None:
        """Append a new pose given the relative motion from previous to current frame.

        The PnP solver returns (R, t) that maps points expressed in the previous
        camera frame into the current camera frame:  X_curr = R @ X_prev + t.
        The inverse, used to evolve the world pose, is R^T, -R^T t.
        """
        R_prev_world = self.rotations[-1]
        p_prev_world = self.positions[-1]
        R_curr_world = R_prev_world @ R_prev_to_curr.T
        p_curr_world = p_prev_world - R_curr_world @ t_in_prev
        self.rotations.append(R_curr_world)
        self.positions.append(p_curr_world)

    @property
    def length_m(self) -> float:
        pts = np.asarray(self.positions)
        if len(pts) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    @property
    def endpoint_m(self) -> np.ndarray:
        return self.positions[-1].copy()


def detect_features(gray: np.ndarray, max_features: int = 2000) -> FrameFeatures:
    orb = cv2.ORB_create(nfeatures=max_features)
    kps, desc = orb.detectAndCompute(gray, None)
    if desc is None:
        desc = np.zeros((0, 32), dtype=np.uint8)
    return FrameFeatures(keypoints=list(kps), descriptors=desc)


def match_features(
    a: FrameFeatures, b: FrameFeatures, ratio: float = 0.75
) -> list[cv2.DMatch]:
    if len(a.descriptors) == 0 or len(b.descriptors) == 0:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(a.descriptors, b.descriptors, k=2)
    good: list[cv2.DMatch] = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def backproject(uv: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Lift pixel coordinates + per-pixel depth into 3D camera-frame points.

    `uv` is (N, 2), `depth` is (H, W), K is (3, 3). Returns (N, 3) in meters.
    Pixels with non-finite or non-positive depth get NaN and should be filtered.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = uv[:, 0]
    v = uv[:, 1]
    H, W = depth.shape
    u_i = np.clip(np.round(u).astype(int), 0, W - 1)
    v_i = np.clip(np.round(v).astype(int), 0, H - 1)
    z = depth[v_i, u_i].astype(np.float64)
    bad = ~np.isfinite(z) | (z <= 0)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    out = np.stack([x, y, z], axis=-1)
    out[bad] = np.nan
    return out


def estimate_relative_pose(
    pts3d_prev: np.ndarray,
    uv_curr: np.ndarray,
    K: np.ndarray,
) -> StepResult:
    """Solve PnP with RANSAC for the relative pose previous→current frame.

    Returns a StepResult. NaN 3D points are dropped before solving.
    """
    mask = np.all(np.isfinite(pts3d_prev), axis=1)
    pts3d = pts3d_prev[mask].astype(np.float32)
    uv = uv_curr[mask].astype(np.float32)

    num_matches = len(pts3d_prev)
    if len(pts3d) < 6:
        return StepResult(R=np.eye(3), t=np.zeros(3), num_inliers=0, num_matches=num_matches)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d,
        uv,
        K,
        None,
        iterationsCount=200,
        reprojectionError=3.0,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inliers is None or len(inliers) < 6:
        return StepResult(R=np.eye(3), t=np.zeros(3), num_inliers=0, num_matches=num_matches)

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    return StepResult(R=R, t=t, num_inliers=int(len(inliers)), num_matches=num_matches)


def step(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_depth: np.ndarray,
    K: np.ndarray,
) -> StepResult:
    """Run one VO step between two consecutive frames given the depth of the previous one."""
    f_prev = detect_features(prev_gray)
    f_curr = detect_features(curr_gray)
    matches = match_features(f_prev, f_curr)
    if len(matches) < 6:
        return StepResult(R=np.eye(3), t=np.zeros(3), num_inliers=0, num_matches=len(matches))
    uv_prev = np.array([f_prev.keypoints[m.queryIdx].pt for m in matches], dtype=np.float64)
    uv_curr = np.array([f_curr.keypoints[m.trainIdx].pt for m in matches], dtype=np.float64)
    pts3d_prev = backproject(uv_prev, prev_depth, K)
    return estimate_relative_pose(pts3d_prev, uv_curr, K)
