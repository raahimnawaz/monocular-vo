"""Render the VO vs SLAM vs ground-truth top-down comparison plot.

Reads the .npy outputs produced by run_tum.py (or any run that saves
`vo_positions.npy`, `slam_positions.npy`, `gt_keyframe_positions.npy`).
Aligns each estimate to GT with Umeyama (so we're comparing trajectory
*shape* fairly given monocular scale drift).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from monocular_vo.eval import umeyama  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", type=Path, required=True)
    ap.add_argument("--save", type=Path, required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    vo = np.load(args.outputs / "vo_positions.npy")
    slam = np.load(args.outputs / "slam_positions.npy")
    gt = np.load(args.outputs / "gt_keyframe_positions.npy")

    # SLAM positions are per-keyframe (same count as gt); VO positions are per-step
    # and longer. Trim VO to keyframe count for fair shape comparison.
    n_kf = len(gt)
    vo_aligned_src = vo[:n_kf] if len(vo) >= n_kf else vo
    gt_for_vo = gt[: len(vo_aligned_src)]

    t_vo = umeyama(vo_aligned_src, gt_for_vo, with_scale=True)
    vo_a = t_vo.apply(vo_aligned_src)
    t_slam = umeyama(slam, gt, with_scale=True)
    slam_a = t_slam.apply(slam)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"wspace": 0.25})

    def _topdown(ax, vo_xy, slam_xy, gt_xy, lc_drop: bool, label: str):
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], "-", color="black", linewidth=1.6, label="ground truth")
        if not lc_drop:
            ax.plot(vo_xy[:, 0], vo_xy[:, 1], "-", color="#d62728", linewidth=1.2, alpha=0.9, label="VO (no LC)")
        else:
            ax.plot(slam_xy[:, 0], slam_xy[:, 1], "-", color="#1f77b4", linewidth=1.4, alpha=0.95, label="VO + pose-graph + LC")
        ax.scatter(gt_xy[0, 0], gt_xy[0, 1], c="g", s=50, zorder=5, label="start")
        ax.set_xlabel("X (m, world)")
        ax.set_ylabel("Y (m, world)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(label)
        ax.legend(loc="best", fontsize=9)

    _topdown(axes[0], vo_a[:, [0, 1]], None, gt_for_vo[:, [0, 1]], False, "VO only")
    _topdown(axes[1], None, slam_a[:, [0, 1]], gt[:, [0, 1]], True, "VO + pose-graph + loop closure")

    if args.title:
        fig.suptitle(args.title, fontsize=13, y=1.02)
    fig.savefig(args.save, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.save}")


if __name__ == "__main__":
    main()
