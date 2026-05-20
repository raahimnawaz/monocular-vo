"""Interactive chessboard calibration.

1. Hold the printed chessboard so the corners are fully visible.
2. SPACE captures a view. Repeat ~20 times from varied angles + distances.
3. Press q when done — calibration runs and is saved to --output.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from monocular_vo.calibrate import calibrate_from_image_paths
from monocular_vo.capture import interactive_chessboard_capture


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("data/calibration.npz"))
    ap.add_argument("--image-dir", type=Path, default=Path("data/chessboard"))
    ap.add_argument("--target-views", type=int, default=20)
    ap.add_argument(
        "--skip-capture",
        action="store_true",
        help="reuse existing chessboard images in --image-dir without opening the camera",
    )
    args = ap.parse_args()

    if not args.skip_capture:
        saved = interactive_chessboard_capture(args.image_dir, args.target_views)
        if saved < 10:
            raise SystemExit(f"only {saved} views captured; need >= 10")

    image_paths = sorted(args.image_dir.glob("chessboard_*.png"))
    print(f"calibrating from {len(image_paths)} images")
    calib = calibrate_from_image_paths(image_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calib.save(args.output)

    print("--- calibration result ---")
    print("K =")
    print(calib.K)
    print("dist =", calib.dist.ravel())
    print(f"image_size = {calib.image_size}")
    print(f"reprojection_error = {calib.reprojection_error:.4f} px")
    if calib.reprojection_error > 0.5:
        print("WARNING: reprojection error > 0.5 px — recapture from more diverse angles")
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
