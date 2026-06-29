"""Pairwise top-K pseudo-label candidate selection."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def select_pairwise_topk(
    probs: np.ndarray,
    sample_ids: list[str],
    class_pairs: list[tuple[int, int]],
    top_k_per_pair: int,
    min_pair_score: float,
    exclude_sample_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Select pseudo-label candidates independently per class pair.

    Pair score is ``min(prob_a, prob_b)``. The function intentionally performs
    no global top-K operation; each pair receives its own ranking and cap.
    """
    prob_array = np.asarray(probs, dtype=float)
    if prob_array.ndim != 2:
        raise ValueError("probs must be a 2D array of shape [n_samples, n_classes]")
    if len(sample_ids) != prob_array.shape[0]:
        raise ValueError("sample_ids length must match probs rows")
    if top_k_per_pair <= 0:
        raise ValueError("top_k_per_pair must be positive")

    excluded = set(exclude_sample_ids or set())
    rows: list[dict[str, object]] = []
    for class_a, class_b in _iter_valid_pairs(class_pairs, prob_array.shape[1]):
        pair_rows: list[dict[str, object]] = []
        for row_idx, sample_id in enumerate(sample_ids):
            if sample_id in excluded:
                continue
            prob_a = float(prob_array[row_idx, class_a])
            prob_b = float(prob_array[row_idx, class_b])
            pair_score = min(prob_a, prob_b)
            if pair_score < min_pair_score:
                continue
            pair_rows.append(
                {
                    "sample_id": sample_id,
                    "class_a": class_a,
                    "class_b": class_b,
                    "prob_a": prob_a,
                    "prob_b": prob_b,
                    "pair_score": pair_score,
                }
            )
        pair_rows.sort(key=lambda row: (-float(row["pair_score"]), str(row["sample_id"])))
        for rank, row in enumerate(pair_rows[:top_k_per_pair], start=1):
            row["rank_within_pair"] = rank
            rows.append(row)

    columns = ["sample_id", "class_a", "class_b", "prob_a", "prob_b", "pair_score", "rank_within_pair"]
    return pd.DataFrame(rows, columns=columns)


def _iter_valid_pairs(class_pairs: Iterable[tuple[int, int]], n_classes: int) -> Iterable[tuple[int, int]]:
    for class_a, class_b in class_pairs:
        if class_a == class_b:
            raise ValueError("class pair entries must refer to two different classes")
        if not (0 <= class_a < n_classes and 0 <= class_b < n_classes):
            raise ValueError(f"class pair {(class_a, class_b)} is outside n_classes={n_classes}")
        yield int(class_a), int(class_b)
