"""Trajectory plotting + side-by-side animation export."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def plot_trajectory(
    positions: np.ndarray, output_path: Path, ground_truth_length: float | None = None
) -> None:
    """Render a 3D trajectory PNG.

    `positions` is (N, 3) in meters.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0], positions[:, 2], positions[:, 1], "-", linewidth=1.5)
    ax.scatter(*positions[0, [0, 2, 1]], c="g", s=40, label="start")
    ax.scatter(*positions[-1, [0, 2, 1]], c="r", s=40, label="end")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m, forward)")
    ax.set_zlabel("Y (m, vertical)")
    title = "Estimated trajectory"
    if ground_truth_length is not None:
        title += f"  (ground-truth length: {ground_truth_length:.2f} m)"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
