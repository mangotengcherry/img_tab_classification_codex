"""FBM failure-pattern domain knowledge, grounded in the literature.

Reference
---------
B. Kim, Y.-S. Jeong, S. H. Tong, I.-K. Chang, M.-K. Jeong,
"A Regularized Singular Value Decomposition-based Approach for Failure Pattern
Classification on Fail Bit Map in a DRAM Wafer," IEEE Trans. Semiconductor
Manufacturing, vol. 28, no. 1, 2015 (papers/RSVD.pdf).

Domain facts used here (see docs/fbm_domain_notes.md):

1. FBM grade (0..N) is a NONLINEAR severity of failed-cell count, not a linear
   count. Grade 0 = no fail; high grade = many failed cells.
2. Structured failure patterns live in HIGH grades. The paper binarizes an FBM
   with the engineering rule ``grade >= 3 -> 1`` (Eq. 7) before pattern analysis.
3. Two physical families:
   - single-bit  : spatially RANDOM, LOW grade (cell-area, no structure)
   - non-single-bit : spatially STRUCTURED, HIGH grade - vertical line, horizontal
     line, cluster (core/periphery: row/column drivers, sense-amps).
   Physical mapping: horizontal line ~ word-line(row), vertical line ~ bit-line
   (column), single-bit ~ random cell leakage.
4. Discriminative features = eigen-image Frobenius norms of the (R)SVD of the
   binarized FBM (Eq. 8: ||E_k|| = s_kk). Structured maps are low-rank (energy in
   a few eigen-images, fast norm decay); random maps are high-rank (slow decay).

This module provides the binarization, the eigen-image-norm features, a
nonnegative factorization (Lee-Seung multiplicative updates - the algorithm
family the paper's RSVD extends), and canonical pattern painters used by the
synthetic data generator.
"""

from __future__ import annotations

import numpy as np

IMAGE_SHAPE = (128, 46)
MAX_GRADE = 8.0
BINARIZE_THRESHOLD = 3.0  # paper Eq. 7: grade >= 3 is a "fail" unit block

# Word-line bands over the 128 image rows (top/middle/bottom). In reality this
# comes from measurement_map.csv; here it grounds the top/bottom identity split.
TOP_ROWS = slice(0, 43)
MIDDLE_ROWS = slice(43, 85)
BOTTOM_ROWS = slice(85, 128)


# --------------------------------------------------------------------------
# binarization + eigen-image-norm features (paper Section IV.A / Eq. 8)
# --------------------------------------------------------------------------
def binarize_fbm(image: np.ndarray, threshold: float = BINARIZE_THRESHOLD) -> np.ndarray:
    """Binarize an FBM grade image (paper Eq. 7): 1 where grade >= threshold."""
    return (np.asarray(image, dtype=float) >= threshold).astype(float)


def _singular_values(binary_image: np.ndarray) -> np.ndarray:
    """Singular values (descending) via the smaller Gram matrix + eigvalsh.

    Equivalent to the singular values of the matrix but numerically robust: LAPACK
    gesdd (np.linalg.svd) can return NaN on some sparse binary maps, whereas the
    symmetric eigensolver on B^T B (or B B^T) is stable.
    """
    arr = np.asarray(binary_image, dtype=float)
    if arr.ndim != 2:
        raise ValueError("binary_image must be 2D")
    m, n = arr.shape
    gram = arr.T @ arr if n <= m else arr @ arr.T
    eigvals = np.linalg.eigvalsh(gram)        # ascending, real (PSD)
    sv = np.sqrt(np.clip(eigvals[::-1], 0.0, None))
    return sv


def eigenimage_norm_features(binary_image: np.ndarray, k: int = 16) -> np.ndarray:
    """First ``k`` eigen-image Frobenius norms of a binarized FBM (paper Eq. 8).

    The singular values of the binary matrix equal the eigen-image norms
    ``||E_k|| = s_kk``. Structured patterns (line/cluster) concentrate energy in a
    few eigen-images; random single-bit maps spread it across many. Returned in
    descending order, zero-padded to length ``k``.
    """
    sv = _singular_values(binary_image)
    out = np.zeros(k, dtype=float)
    out[: min(k, sv.shape[0])] = sv[:k]
    return out


def effective_rank(binary_image: np.ndarray, energy: float = 0.9) -> int:
    """Number of eigen-images needed to capture ``energy`` of the squared norm.

    Low for structured (line/cluster) maps, high for random single-bit maps -
    a compact scalar version of the paper's Fig. 6 norm-decay discriminator.
    """
    sv = _singular_values(binary_image)
    total = float((sv ** 2).sum())
    if total == 0.0:
        return 0
    cumulative = np.cumsum(sv ** 2) / total
    return int(np.searchsorted(cumulative, energy) + 1)


def nonneg_factorize(
    binary_image: np.ndarray, k: int = 16, n_iter: int = 60, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank-k nonnegative factorization X ~= W H (Lee-Seung multiplicative updates).

    Initialized from |SVD| as in the paper's RSVD Step 1. This is the nonnegative
    factorization family the paper's RSVD extends (with an added binary penalty);
    we keep the robust Lee-Seung form. Returns (W, H, component_norms) with
    ``component_norms`` = Frobenius norm of each rank-1 term ``W[:,r] H[r,:]``,
    sorted descending.
    """
    x = np.asarray(binary_image, dtype=float)
    m, n = x.shape
    k = min(k, m, n)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    rng = np.random.default_rng(seed)
    w = np.abs(u[:, :k] * np.sqrt(s[:k])) + 1e-6
    h = np.abs((vt[:k, :].T * np.sqrt(s[:k])).T) + 1e-6
    eps = 1e-9
    for _ in range(n_iter):
        h *= (w.T @ x) / (w.T @ w @ h + eps)
        w *= (x @ h.T) / (w @ (h @ h.T) + eps)
    norms = np.array([np.linalg.norm(np.outer(w[:, r], h[r, :])) for r in range(k)])
    order = np.argsort(norms)[::-1]
    return w[:, order], h[order, :], norms[order]


def image_feature_matrix(
    images: np.ndarray, *, pool: int = 2, k_eigen: int = 16, max_grade: float = MAX_GRADE
) -> np.ndarray:
    """Assemble paper-grounded image features for the fusion model.

    Per image, concatenates:
      - normalized graded FBM, average-pooled (grade intensity),
      - binarized (grade>=3) map, average-pooled (structured-pattern channel),
      - first ``k_eigen`` eigen-image norms of the binarized map (global structure).
    """
    images = np.asarray(images, dtype=float)
    graded = _avg_pool(images / max_grade, pool).reshape(images.shape[0], -1)
    binary = np.stack([binarize_fbm(img) for img in images])
    binary_pooled = _avg_pool(binary, pool).reshape(images.shape[0], -1)
    norm_scale = np.sqrt(images.shape[1] * images.shape[2])
    eigen = np.stack([eigenimage_norm_features(b, k=k_eigen) for b in binary]) / norm_scale
    return np.concatenate([graded, binary_pooled, eigen], axis=1)


def _avg_pool(images: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return images
    n, h, w = images.shape
    h2, w2 = h // factor, w // factor
    trimmed = images[:, : h2 * factor, : w2 * factor]
    return trimmed.reshape(n, h2, factor, w2, factor).mean(axis=(2, 4))


# --------------------------------------------------------------------------
# canonical pattern painters (paper Section II / Fig. 3)
# --------------------------------------------------------------------------
def paint_single_bit_scatter(
    img: np.ndarray, rng: np.random.Generator, *, density: float = 0.015, grade_range: tuple[int, int] = (1, 2)
) -> None:
    """Random LOW-grade single-bit fails (paper: single-bit = random, low grade).

    Mutates ``img`` in place. These sit below the grade>=3 binarization threshold,
    so they are the "noise" that structured patterns must be separated from.
    """
    h, w = img.shape
    n = int(density * h * w)
    if n <= 0:
        return
    rows = rng.integers(0, h, size=n)
    cols = rng.integers(0, w, size=n)
    img[rows, cols] += rng.uniform(grade_range[0], grade_range[1] + 1, size=n)


def paint_vertical_line(img: np.ndarray, *, col: int, width: int = 2, grade: float = 6.0) -> None:
    """High-grade vertical line ~ bit-line / column failure (structured, non-single-bit)."""
    w = img.shape[1]
    lo = max(0, col - width // 2)
    hi = min(w, lo + width)
    img[:, lo:hi] += grade


def paint_horizontal_line(img: np.ndarray, *, rows: slice, height: int = 3, grade: float = 6.0) -> None:
    """High-grade horizontal line ~ word-line / row failure (structured)."""
    start = rows.start or 0
    img[start : start + height, :] += grade


def paint_cluster(img: np.ndarray, *, center: tuple[int, int], half: tuple[int, int] = (12, 8), grade: float = 6.5) -> None:
    """High-grade block / cluster (structured, localized)."""
    cy, cx = center
    hy, hx = half
    img[max(0, cy - hy) : cy + hy, max(0, cx - hx) : cx + hx] += grade


def paint_edge_ring(img: np.ndarray, *, width: int = 5, grade: float = 6.0) -> None:
    """High-grade peripheral ring (edge / periphery failure)."""
    img[:width, :] += grade
    img[-width:, :] += grade
    img[:, :3] += grade
    img[:, -3:] += grade
