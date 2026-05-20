"""Trajectory evaluation: ATE and RPE following the conventions of Sturm et al.
(IROS 2012) and the KITTI odometry benchmark.

Two estimated and ground-truth trajectories are aligned with a similarity
transform (Umeyama 1991) — this is essential for monocular VO whose scale
can drift even when a depth model gives us metric units.

Metrics:
- ATE (Absolute Trajectory Error): RMSE of per-pose position residuals after
  alignment. Reported in meters.
- RPE (Relative Pose Error): RMSE of frame-to-frame translation deltas after
  alignment. Reported in meters per frame (or per segment, with `delta`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UmeyamaResult:
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)
    scale: float

    def apply(self, points: np.ndarray) -> np.ndarray:
        return (self.scale * (self.R @ points.T)).T + self.t


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> UmeyamaResult:
    """Find similarity (R, t, s) that best maps src -> dst in least-squares sense.

    Both arrays are (N, 3). Implements Umeyama 1991. When `with_scale=False`
    the scale is fixed to 1.0 (rigid alignment).
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt

    if with_scale:
        var_src = (src_c**2).sum() / n
        scale = (D * np.diag(S)).sum() / var_src if var_src > 0 else 1.0
    else:
        scale = 1.0
    t = mu_dst - scale * R @ mu_src
    return UmeyamaResult(R=R, t=t, scale=float(scale))


def ate_rmse(est: np.ndarray, gt: np.ndarray, with_scale: bool = True) -> float:
    """Absolute trajectory error (RMSE in meters) after similarity alignment."""
    assert est.shape == gt.shape and est.shape[1] == 3 and len(est) >= 3
    transform = umeyama(est, gt, with_scale=with_scale)
    aligned = transform.apply(est)
    err = np.linalg.norm(aligned - gt, axis=1)
    return float(np.sqrt((err**2).mean()))


def rpe_translation(
    est: np.ndarray,
    gt: np.ndarray,
    delta: int = 1,
    with_scale: bool = True,
) -> float:
    """Relative pose error (translation RMSE in meters) over windows of `delta` frames.

    For each pair (i, i+delta), measure the per-step translation in both the
    estimated and ground-truth trajectory after alignment, and report the RMSE
    of their difference.
    """
    assert est.shape == gt.shape and len(est) > delta
    transform = umeyama(est, gt, with_scale=with_scale)
    aligned = transform.apply(est)
    deltas_est = aligned[delta:] - aligned[:-delta]
    deltas_gt = gt[delta:] - gt[:-delta]
    err = np.linalg.norm(deltas_est - deltas_gt, axis=1)
    return float(np.sqrt((err**2).mean()))


def scale_error(est: np.ndarray, gt: np.ndarray) -> float:
    """Path-length ratio as a percentage: 100 * (len(est) - len(gt)) / len(gt)."""
    if len(est) < 2 or len(gt) < 2:
        return 0.0
    len_est = float(np.linalg.norm(np.diff(est, axis=0), axis=1).sum())
    len_gt = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
    if len_gt == 0:
        return 0.0
    return 100.0 * (len_est - len_gt) / len_gt
