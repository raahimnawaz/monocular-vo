"""Chessboard camera calibration using OpenCV.

Captures live frames from the webcam, detects a `BOARD_SIZE` chessboard, and
runs `cv2.calibrateCamera` once enough views are collected. The resulting
intrinsics matrix `K` and distortion coefficients are saved to disk as a .npz.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

BOARD_SIZE = (9, 6)
SQUARE_SIZE_M = 0.025


@dataclass
class Calibration:
    K: np.ndarray
    dist: np.ndarray
    image_size: tuple[int, int]
    reprojection_error: float

    def save(self, path: Path) -> None:
        np.savez(
            path,
            K=self.K,
            dist=self.dist,
            image_size=np.array(self.image_size),
            reprojection_error=np.array(self.reprojection_error),
        )

    @classmethod
    def load(cls, path: Path) -> Calibration:
        data = np.load(path)
        return cls(
            K=data["K"],
            dist=data["dist"],
            image_size=tuple(int(x) for x in data["image_size"]),
            reprojection_error=float(data["reprojection_error"]),
        )


def _object_points_template() -> np.ndarray:
    """3D coordinates of the chessboard corners in the board's own frame."""
    cols, rows = BOARD_SIZE
    pts = np.zeros((cols * rows, 3), dtype=np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    pts *= SQUARE_SIZE_M
    return pts


def calibrate_from_image_paths(image_paths: list[Path]) -> Calibration:
    """Run calibration over a list of already-captured chessboard images."""
    object_pts: list[np.ndarray] = []
    image_pts: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    template = _object_points_template()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])
        found, corners = cv2.findChessboardCorners(img, BOARD_SIZE, None)
        if not found:
            continue
        cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        object_pts.append(template)
        image_pts.append(corners)

    if len(image_pts) < 10:
        raise RuntimeError(
            f"only {len(image_pts)} usable chessboard views — need >= 10 for stable calibration"
        )
    assert image_size is not None

    rep_err, K, dist, _, _ = cv2.calibrateCamera(
        object_pts, image_pts, image_size, None, None
    )
    return Calibration(
        K=K,
        dist=dist,
        image_size=image_size,
        reprojection_error=float(rep_err),
    )
