"""Generate a side-by-side demo GIF: input frame | depth heatmap | trajectory.

Re-runs the VO pipeline on a captured video, saves a composite image per step,
and stitches them into a GIF suitable for README embedding.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from tqdm import tqdm  # noqa: E402

from monocular_vo.calibrate import Calibration  # noqa: E402
from monocular_vo.depth import DEFAULT_MODEL, DepthAnything  # noqa: E402
from monocular_vo.vo import Trajectory, step  # noqa: E402


PANEL_W = 480
PANEL_H = 270


def _resize(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)


def _depth_to_heatmap(depth: np.ndarray) -> np.ndarray:
    """Map a metric depth map to a uint8 BGR heatmap for display."""
    finite = depth[np.isfinite(depth)]
    if len(finite) == 0:
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    lo = float(np.percentile(finite, 2.0))
    hi = float(np.percentile(finite, 98.0))
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    inverted = 1.0 - norm  # closer = warmer
    coloured = cv2.applyColorMap((inverted * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return coloured


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (PANEL_W, 36), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return out


def _trajectory_panel(
    fig: Figure,
    positions: np.ndarray,
    ground_truth_length: float | None,
) -> np.ndarray:
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(positions[:, 0], positions[:, 2], "-", color="#1f77b4", linewidth=2)
    ax.plot(positions[0, 0], positions[0, 2], "o", color="green", markersize=8, label="start")
    ax.plot(positions[-1, 0], positions[-1, 2], "o", color="red", markersize=8, label="current")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m, forward)")
    span = float(max(np.ptp(positions[:, 0]), np.ptp(positions[:, 2]), 1.0))
    cx, cz = positions[:, 0].mean(), positions[:, 2].mean()
    pad = max(span / 2.0 + 0.5, 1.5)
    ax.set_xlim(cx - pad, cx + pad)
    ax.set_ylim(cz - pad, cz + pad)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    length = 0.0
    if len(positions) > 1:
        length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    title = f"trajectory  ({length:.2f} m"
    if ground_truth_length is not None:
        title += f" / gt {ground_truth_length:.2f} m"
    title += ")"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    return _resize(bgr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--calibration", type=Path, default=Path("data/calibration.npz"))
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("figures/demo.mp4"),
        help="output path; .mp4 (recommended) or .gif",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=4,
        help="process every Nth frame (higher = shorter, lighter GIF)",
    )
    ap.add_argument("--fps", type=int, default=10, help="output GIF framerate")
    ap.add_argument("--depth-model", default=DEFAULT_MODEL)
    ap.add_argument("--ground-truth-length", type=float, default=None)
    ap.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="cap on composite frames written (keeps GIF size manageable)",
    )
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"no such video: {args.video}")
    if not args.calibration.exists():
        raise SystemExit(f"no such calibration: {args.calibration}")

    calib = Calibration.load(args.calibration)
    K = calib.K
    depth_model = DepthAnything(model_id=args.depth_model)
    print(f"depth device: {depth_model.device}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")

    trajectory = Trajectory()
    prev_gray: np.ndarray | None = None
    prev_depth: np.ndarray | None = None
    composites: list[np.ndarray] = []
    fig = Figure(figsize=(PANEL_W / 100, PANEL_H / 100), dpi=100)
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(fig)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    idx = 0
    written = 0
    pbar = tqdm(desc="demo")
    try:
        while written < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % args.stride != 0:
                idx += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            depth = depth_model.predict(frame)

            if prev_gray is not None and prev_depth is not None:
                result = step(prev_gray, gray, prev_depth, K)
                if not result.degenerate:
                    trajectory.update(result.R, result.t)

            positions = np.asarray(trajectory.positions)
            frame_panel = _label(_resize(frame), "input")
            depth_panel = _label(_resize(_depth_to_heatmap(depth)), "Depth Anything v2 (metric)")
            traj_panel = _label(
                _trajectory_panel(fig, positions, args.ground_truth_length),
                "metric trajectory (top-down)",
            )
            composite = np.concatenate([frame_panel, depth_panel, traj_panel], axis=1)
            composites.append(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
            written += 1

            prev_gray = gray
            prev_depth = depth
            idx += 1
            pbar.update(1)
    finally:
        cap.release()
        pbar.close()

    if not composites:
        raise SystemExit("no frames produced")
    print(f"writing {len(composites)} frames to {args.output}")
    suffix = args.output.suffix.lower()
    if suffix in (".mp4", ".webm", ".mov"):
        # H.264 via imageio-ffmpeg's bundled binary; pad to even dims if needed.
        h, w = composites[0].shape[:2]
        if h % 2 or w % 2:
            composites = [
                f[: (h // 2) * 2, : (w // 2) * 2] for f in composites
            ]
        writer = imageio.get_writer(
            args.output,
            fps=args.fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=1,
        )
        try:
            for frame in composites:
                writer.append_data(frame)
        finally:
            writer.close()
    else:
        # GIF path: route via Pillow with palette quantization for ~2-3x smaller output.
        from PIL import Image

        duration_ms = int(1000 / args.fps)
        # Build a global palette from one representative frame so all frames share it
        # (smaller file + smoother color transitions than per-frame palettes).
        ref = Image.fromarray(composites[len(composites) // 2]).quantize(
            colors=64, method=Image.Quantize.MEDIANCUT
        )
        palette = ref.getpalette()
        ref_with_palette = Image.new("P", ref.size)
        ref_with_palette.putpalette(palette)
        pil_frames = [
            Image.fromarray(f).quantize(palette=ref_with_palette, dither=Image.Dither.FLOYDSTEINBERG)
            for f in composites
        ]
        pil_frames[0].save(
            args.output,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
            disposal=2,
        )
    print(f"done: {args.output}  ({args.output.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
