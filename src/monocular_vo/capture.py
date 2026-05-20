"""Webcam capture helpers.

Two modes:
- `record_video`: open a `VideoCapture`, save raw frames to an mp4 at native FPS
- `interactive_chessboard_capture`: live preview; press SPACE to grab a frame
  when a chessboard is visible
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2

from .calibrate import BOARD_SIZE


def open_camera(device: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera device {device}")
    # Encourage a stable resolution; macOS will pick the closest supported.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def record_video(
    output_path: Path,
    duration_s: float,
    device: int = 0,
    show_preview: bool = True,
) -> tuple[int, int, float]:
    """Record `duration_s` seconds of webcam footage to `output_path`.

    Returns (width, height, fps_estimated).
    """
    cap = open_camera(device)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_hint = cap.get(cv2.CAP_PROP_FPS) or 30.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps_hint, (width, height))

    print(f"recording {duration_s:.1f}s at {width}x{height} -> {output_path}")
    print("press 'q' to stop early")

    t0 = time.time()
    frame_count = 0
    try:
        while time.time() - t0 < duration_s:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            frame_count += 1
            if show_preview:
                cv2.imshow("capture", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        if show_preview:
            cv2.destroyAllWindows()

    elapsed = time.time() - t0
    measured_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"recorded {frame_count} frames in {elapsed:.1f}s -> {measured_fps:.1f} fps measured")
    return width, height, measured_fps


def interactive_chessboard_capture(output_dir: Path, target_views: int = 20) -> int:
    """Live preview; SPACE captures a frame when a chessboard is detected.

    Returns the number of saved images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = open_camera()
    print(f"need {target_views} chessboard views — move the board between captures")
    print("SPACE = grab when corners highlight; q = quit")

    saved = 0
    try:
        while saved < target_views:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)
            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, BOARD_SIZE, corners, found)
            cv2.putText(
                display,
                f"{saved}/{target_views}  {'OK' if found else 'searching...'}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0) if found else (0, 0, 255),
                2,
            )
            cv2.imshow("calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and found:
                path = output_dir / f"chessboard_{saved:02d}.png"
                cv2.imwrite(str(path), frame)
                saved += 1
                print(f"saved {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return saved
