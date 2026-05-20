"""KITTI Odometry benchmark loader.

Expected layout (the standard from the KITTI download):

    <root>/sequences/<seq>/image_2/000000.png   # left color frames
    <root>/sequences/<seq>/calib.txt            # 3x4 projection matrices P0..P3
    <root>/poses/<seq>.txt                      # ground-truth poses (sequences 00-10)

For monocular VO we only need:
    image_2 (the left RGB camera, since it's higher resolution than image_0/1)
    P2 from calib.txt (3x4 projection matrix for cam2)
    poses/<seq>.txt (only the camera-origin positions are used; we ignore the
        full 6-DoF GT pose because monocular VO is up to similarity anyway).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class KittiSequence:
    root: Path
    seq_id: str
    K: np.ndarray
    image_paths: list[Path]
    gt_positions: np.ndarray  # (N, 3) camera origin in world coords

    def __len__(self) -> int:
        return len(self.image_paths)

    def load_frame(self, i: int) -> np.ndarray:
        img = cv2.imread(str(self.image_paths[i]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"could not read {self.image_paths[i]}")
        return img


def _parse_calib(path: Path, key: str = "P2") -> np.ndarray:
    """Parse a KITTI calib.txt and return the 3x3 intrinsics for the chosen camera.

    KITTI projection matrices are 3x4; the rectified intrinsics K are the
    first three columns (since left-camera origin coincides with rectified
    world origin for P0/P2).
    """
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        name, _, rest = line.partition(":")
        if name.strip() == key:
            vals = np.array([float(x) for x in rest.split()]).reshape(3, 4)
            return vals[:, :3].copy()
    raise KeyError(f"{key} not found in {path}")


def _parse_poses(path: Path) -> np.ndarray:
    """Parse KITTI ground-truth poses file (rows of 12 floats = 3x4 [R|t])."""
    rows = []
    for line in path.read_text().splitlines():
        vals = [float(x) for x in line.split()]
        if len(vals) != 12:
            continue
        rows.append(np.array(vals).reshape(3, 4))
    if not rows:
        raise RuntimeError(f"no poses parsed from {path}")
    # Camera-origin position in world coords is the last column.
    positions = np.array([T[:, 3] for T in rows])
    return positions


def load(root: Path, seq_id: str) -> KittiSequence:
    """Load a KITTI odometry sequence.

    `root` should be the directory containing `sequences/` and `poses/`.
    """
    seq_dir = root / "sequences" / seq_id
    image_dir = seq_dir / "image_2"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"no image_2 directory: {image_dir}")

    K = _parse_calib(seq_dir / "calib.txt", key="P2")
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"no png frames in {image_dir}")

    poses_path = root / "poses" / f"{seq_id}.txt"
    if poses_path.exists():
        gt_positions = _parse_poses(poses_path)
    else:
        gt_positions = np.zeros((len(image_paths), 3))
        print(f"warning: no GT poses at {poses_path}; metrics will be unavailable")

    return KittiSequence(
        root=root, seq_id=seq_id, K=K, image_paths=image_paths, gt_positions=gt_positions
    )
