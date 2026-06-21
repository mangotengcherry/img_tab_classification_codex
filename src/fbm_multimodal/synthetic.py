from __future__ import annotations

import numpy as np


def compose_fbm_images(
    first: np.ndarray,
    second: np.ndarray,
    *,
    mode: str,
    alpha: float = 1.0,
    max_grade: float = 8.0,
) -> np.ndarray:
    """Compose two graded FBM images with an explicit synthetic overlay rule."""
    left = np.asarray(first, dtype=np.float32)
    right = np.asarray(second, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError(f"FBM shapes must match; got {left.shape} and {right.shape}")

    if mode == "max":
        composed = np.maximum(left, right)
    elif mode == "clipped_sum":
        composed = left + right
    elif mode == "weighted_saturating_sum":
        composed = left + alpha * right
    else:
        raise ValueError(f"unknown composition mode: {mode}")

    return np.clip(composed, 0.0, max_grade)


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float32).reshape(-1)
    right = np.asarray(second, dtype=np.float32).reshape(-1)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    value = float(np.dot(left, right) / (left_norm * right_norm))
    return float(np.clip(value, 0.0, 1.0))
