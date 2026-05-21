"""TUM RGB-D Benchmark loader (Freiburg sequences).

Reference: https://cvg.cit.tum.de/data/datasets/rgbd-dataset

Expected layout under the sequence root:
    rgb.txt              # `<timestamp> rgb/<timestamp>.png` lines
    rgb/*.png            # RGB frames (we don't use the Kinect depth)
    groundtruth.txt      # mocap GT, lines: `<ts> tx ty tz qx qy qz qw`

The depth Anything model supplies our own metric depth, so the dataset's
Kinect depth is intentionally unused — this keeps the benchmark a true
*monocular* evaluation. The Kinect calibration nonetheless gives us the
camera intrinsics for free, which we hard-code per sub-dataset.

Ground truth is mocap (camera-to-world): position (tx, ty, tz) in meters,
rotation as quaternion (qx, qy, qz, qw). RGB frames are matched to the
nearest GT timestamp within `match_tolerance_s`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Intrinsics from the TUM benchmark documentation.
# Distortion is NOT applied to RGB images (we use the raw RGB and the K below).
TUM_INTRINSICS: dict[str, np.ndarray] = {
    "freiburg1": np.array([[517.3, 0.0, 318.6], [0.0, 516.5, 255.3], [0.0, 0.0, 1.0]]),
    "freiburg2": np.array([[520.9, 0.0, 325.1], [0.0, 521.0, 249.7], [0.0, 0.0, 1.0]]),
    "freiburg3": np.array([[535.4, 0.0, 320.1], [0.0, 539.2, 247.6], [0.0, 0.0, 1.0]]),
}


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert (qx, qy, qz, qw) to a 3x3 rotation matrix (camera-to-world)."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    R = np.array(
        [
            [
                1.0 - s * (qy * qy + qz * qz),
                s * (qx * qy - qz * qw),
                s * (qx * qz + qy * qw),
            ],
            [
                s * (qx * qy + qz * qw),
                1.0 - s * (qx * qx + qz * qz),
                s * (qy * qz - qx * qw),
            ],
            [
                s * (qx * qz - qy * qw),
                s * (qy * qz + qx * qw),
                1.0 - s * (qx * qx + qy * qy),
            ],
        ]
    )
    return R


def _read_timestamped_lines(path: Path) -> list[tuple[float, list[str]]]:
    rows: list[tuple[float, list[str]]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            t = float(parts[0])
        except ValueError:
            continue
        rows.append((t, parts[1:]))
    return rows


@dataclass
class TumSequence:
    root: Path
    name: str
    K: np.ndarray
    image_paths: list[Path]
    image_timestamps: np.ndarray  # (N,)
    gt_positions: np.ndarray  # (N, 3) matched to image_timestamps
    gt_rotations: np.ndarray  # (N, 3, 3) matched to image_timestamps
    gt_matched_mask: np.ndarray  # (N,) bool — True where a GT pose was matched

    def __len__(self) -> int:
        return len(self.image_paths)

    def load_frame(self, i: int) -> np.ndarray:
        img = cv2.imread(str(self.image_paths[i]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"could not read {self.image_paths[i]}")
        return img


def _detect_intrinsics(root: Path) -> tuple[str, np.ndarray]:
    name = root.name
    for key in TUM_INTRINSICS:
        if key in name:
            return key, TUM_INTRINSICS[key]
    raise ValueError(
        f"could not infer TUM camera from path {root}; "
        f"expected one of {list(TUM_INTRINSICS)} in the directory name"
    )


def load(root: Path, match_tolerance_s: float = 0.02) -> TumSequence:
    """Load a TUM RGB-D sequence. `root` is the unpacked dataset directory."""
    rgb_txt = root / "rgb.txt"
    gt_txt = root / "groundtruth.txt"
    if not rgb_txt.is_file():
        raise FileNotFoundError(rgb_txt)
    if not gt_txt.is_file():
        raise FileNotFoundError(gt_txt)

    name, K = _detect_intrinsics(root)

    rgb_rows = _read_timestamped_lines(rgb_txt)
    image_paths = [root / parts[0] for _, parts in rgb_rows]
    image_timestamps = np.array([t for t, _ in rgb_rows])

    gt_rows = _read_timestamped_lines(gt_txt)
    gt_ts = np.array([t for t, _ in gt_rows])
    gt_poses: list[tuple[np.ndarray, np.ndarray]] = []
    for _, parts in gt_rows:
        vals = [float(x) for x in parts]
        tx, ty, tz, qx, qy, qz, qw = vals
        gt_poses.append((_quat_to_rot(qx, qy, qz, qw), np.array([tx, ty, tz])))

    # For each RGB frame, find the closest GT timestamp.
    matched_R = np.zeros((len(image_timestamps), 3, 3))
    matched_t = np.zeros((len(image_timestamps), 3))
    matched_mask = np.zeros(len(image_timestamps), dtype=bool)
    for i, ts in enumerate(image_timestamps):
        j = int(np.argmin(np.abs(gt_ts - ts)))
        if abs(gt_ts[j] - ts) <= match_tolerance_s:
            matched_R[i] = gt_poses[j][0]
            matched_t[i] = gt_poses[j][1]
            matched_mask[i] = True

    return TumSequence(
        root=root,
        name=name,
        K=K,
        image_paths=image_paths,
        image_timestamps=image_timestamps,
        gt_positions=matched_t,
        gt_rotations=matched_R,
        gt_matched_mask=matched_mask,
    )
