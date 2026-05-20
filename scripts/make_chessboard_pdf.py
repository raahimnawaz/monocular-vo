"""Generate a printable 9x6-inner-corner chessboard PDF (25 mm squares).

A 9x6 *inner corner* board has 10x7 squares. At 25 mm per square that's
250 x 175 mm — fits on A4 with margin.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main() -> None:
    output = Path("data/chessboard/pattern.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)

    cols_squares, rows_squares = 10, 7  # produces 9x6 inner corners
    square_mm = 25.0
    width_mm = cols_squares * square_mm
    height_mm = rows_squares * square_mm

    fig_w_in = width_mm / 25.4
    fig_h_in = height_mm / 25.4
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    ax.set_xlim(0, cols_squares)
    ax.set_ylim(0, rows_squares)
    ax.set_aspect("equal")
    ax.axis("off")

    for r in range(rows_squares):
        for c in range(cols_squares):
            colour = "black" if (r + c) % 2 == 0 else "white"
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=colour, edgecolor="none"))

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output, format="pdf", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"wrote {output} ({cols_squares}x{rows_squares} squares, {square_mm}mm)")
    print("after printing, measure one square with a ruler and adjust SQUARE_SIZE_M")
    print(f"in src/monocular_vo/calibrate.py if it isn't exactly {square_mm}mm.")


if __name__ == "__main__":
    main()
