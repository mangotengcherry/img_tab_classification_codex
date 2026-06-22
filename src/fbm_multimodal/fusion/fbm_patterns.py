"""Canonical FBM defect pattern *shapes* (domain knowledge only).

These painters make the synthetic FBM images resemble the failure shapes that
actually appear on real DRAM fail bit maps:

- random low-grade single-bit scatter (cell-area random fails),
- high-grade vertical line   (~ bit-line / column failure),
- high-grade horizontal line (~ word-line / row failure),
- high-grade cluster / block,
- high-grade edge / periphery ring.

This is the domain element of the FBM literature - what defects look like - and
is used purely to make the synthetic data realistic. We intentionally do not
adopt the paper's classification method; that is a different task from this
image+tabular fusion experiment.
"""

from __future__ import annotations

import numpy as np

IMAGE_SHAPE = (128, 46)
MAX_GRADE = 8.0


def paint_single_bit_scatter(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    density: float = 0.015,
    grade_range: tuple[int, int] = (1, 2),
) -> None:
    """Random LOW-grade single-bit fails (random cell-area leakage). Mutates ``img``."""
    h, w = img.shape
    n = int(density * h * w)
    if n <= 0:
        return
    rows = rng.integers(0, h, size=n)
    cols = rng.integers(0, w, size=n)
    img[rows, cols] += rng.uniform(grade_range[0], grade_range[1] + 1, size=n)


def paint_vertical_line(img: np.ndarray, *, col: int, width: int = 2, grade: float = 6.0) -> None:
    """High-grade vertical line ~ bit-line / column failure."""
    w = img.shape[1]
    lo = max(0, col - width // 2)
    hi = min(w, lo + width)
    img[:, lo:hi] += grade


def paint_horizontal_line(img: np.ndarray, *, rows: slice, height: int = 3, grade: float = 6.0) -> None:
    """High-grade horizontal line ~ word-line / row failure."""
    start = rows.start or 0
    img[start : start + height, :] += grade


def paint_cluster(
    img: np.ndarray, *, center: tuple[int, int], half: tuple[int, int] = (12, 8), grade: float = 6.5
) -> None:
    """High-grade block / cluster (localized failure)."""
    cy, cx = center
    hy, hx = half
    img[max(0, cy - hy) : cy + hy, max(0, cx - hx) : cx + hx] += grade


def paint_edge_ring(img: np.ndarray, *, width: int = 5, grade: float = 6.0) -> None:
    """High-grade peripheral ring (edge / periphery failure)."""
    img[:width, :] += grade
    img[-width:, :] += grade
    img[:, :3] += grade
    img[:, -3:] += grade
