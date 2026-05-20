"""Record a webcam sequence for VO.

Walk a tape-measured straight line (or any path) while recording. Pass the
measured length to `run_vo.py --ground-truth-length` later.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from monocular_vo.capture import record_video


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True, help="filename stem, e.g. hallway-5m")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--out-dir", type=Path, default=Path("data/sequences"))
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    output = args.out_dir / f"{args.label}.mp4"
    width, height, fps = record_video(
        output_path=output,
        duration_s=args.duration,
        show_preview=not args.no_preview,
    )
    print(f"saved {output} ({width}x{height} @ {fps:.1f} fps)")


if __name__ == "__main__":
    main()
