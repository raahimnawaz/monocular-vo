"""Sanity tests for the TUM RGB-D loader.

Constructs a tiny synthetic TUM-format sequence on disk and verifies that
the loader parses timestamps, matches GT to RGB frames, and emits the
expected intrinsics.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from monocular_vo.tum import TUM_INTRINSICS, load


def _make_synthetic_tum(tmp: Path) -> Path:
    root = tmp / "rgbd_dataset_freiburg1_test"
    (root / "rgb").mkdir(parents=True)
    rgb_lines = ["# color images", "# file: synthetic", "# timestamp filename"]
    timestamps = [1000.0, 1000.05, 1000.10]
    for ts in timestamps:
        rel = f"rgb/{ts}.png"
        cv2.imwrite(str(root / rel), np.zeros((10, 10, 3), dtype=np.uint8))
        rgb_lines.append(f"{ts} {rel}")
    (root / "rgb.txt").write_text("\n".join(rgb_lines))

    gt_lines = ["# ground truth", "# file: synthetic", "# timestamp tx ty tz qx qy qz qw"]
    for k, ts in enumerate([999.995, 1000.04, 1000.099]):  # one is +5ms, two within tol
        x = float(k)
        gt_lines.append(f"{ts} {x} 0.0 0.0 0.0 0.0 0.0 1.0")
    (root / "groundtruth.txt").write_text("\n".join(gt_lines))
    return root


def test_loads_synthetic(tmp_path: Path) -> None:
    root = _make_synthetic_tum(tmp_path)
    seq = load(root, match_tolerance_s=0.02)
    assert len(seq) == 3
    assert seq.name == "freiburg1"
    assert np.allclose(seq.K, TUM_INTRINSICS["freiburg1"])
    assert seq.gt_matched_mask.all(), f"matched mask: {seq.gt_matched_mask}"
    # GT positions should be (0, 0, 0), (1, 0, 0), (2, 0, 0)
    assert np.allclose(seq.gt_positions[0], [0, 0, 0])
    assert np.allclose(seq.gt_positions[1], [1, 0, 0])
    assert np.allclose(seq.gt_positions[2], [2, 0, 0])
    # Identity quaternion → identity rotation
    assert np.allclose(seq.gt_rotations[0], np.eye(3))


def test_unknown_camera_raises(tmp_path: Path) -> None:
    bad = tmp_path / "not_a_known_camera"
    bad.mkdir()
    (bad / "rgb.txt").write_text("# empty\n")
    (bad / "groundtruth.txt").write_text("# empty\n")
    try:
        load(bad)
    except ValueError as e:
        assert "freiburg" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown camera")
