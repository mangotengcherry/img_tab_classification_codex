from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MultiLabelMetrics:
    subset_accuracy: float
    hamming_accuracy: float
    per_class: dict[str, dict[str, float]]


def compute_multilabel_metrics(
    y_true: pd.DataFrame,
    y_prob: pd.DataFrame,
    *,
    thresholds: dict[str, float] | float = 0.5,
) -> MultiLabelMetrics:
    columns = list(y_true.columns)
    prob = y_prob[columns]
    threshold_series = _threshold_series(columns, thresholds)
    y_pred = (prob >= threshold_series).astype(int)
    true = y_true[columns].astype(int)

    exact = (y_pred == true).all(axis=1).mean()
    hamming = (y_pred == true).to_numpy(dtype=float).mean()
    per_class = {
        column: _binary_metrics(true[column].to_numpy(), y_pred[column].to_numpy())
        for column in columns
    }
    return MultiLabelMetrics(
        subset_accuracy=float(exact),
        hamming_accuracy=float(hamming),
        per_class=per_class,
    )


def optimize_class_thresholds(
    y_true: pd.DataFrame,
    y_prob: pd.DataFrame,
    *,
    candidates: list[float] | None = None,
) -> dict[str, float]:
    if candidates is None:
        candidates = [round(value, 2) for value in np.linspace(0.05, 0.95, 19)]

    thresholds: dict[str, float] = {}
    for column in y_true.columns:
        best_threshold = candidates[0]
        best_f1 = -1.0
        for threshold in candidates:
            pred = (y_prob[column] >= threshold).astype(int).to_numpy()
            f1 = _binary_metrics(y_true[column].astype(int).to_numpy(), pred)["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        thresholds[column] = float(best_threshold)
    return thresholds


def class_pair_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, dict[str, float]]:
    columns = list(y_true.columns)
    results: dict[str, dict[str, float]] = {}
    true_binary = y_true[columns].astype(int)
    pred_binary = y_pred[columns].astype(int)

    for left, right in combinations(columns, 2):
        pair_mask = (
            (true_binary[left] == 1)
            & (true_binary[right] == 1)
            & (true_binary.sum(axis=1) == 2)
        )
        support = int(pair_mask.sum())
        if support == 0:
            continue
        exact = (pred_binary.loc[pair_mask, columns] == true_binary.loc[pair_mask, columns]).all(axis=1).mean()
        results[f"{left}+{right}"] = {
            "support": float(support),
            "subset_accuracy": float(exact),
        }
    return results


def synthetic_to_real_gap(
    real_pair_metrics: dict[str, dict[str, float]],
    synthetic_pair_metrics: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for pair_name in sorted(set(real_pair_metrics) & set(synthetic_pair_metrics)):
        real_acc = float(real_pair_metrics[pair_name]["subset_accuracy"])
        synthetic_acc = float(synthetic_pair_metrics[pair_name]["subset_accuracy"])
        results[pair_name] = {
            "real_subset_accuracy": real_acc,
            "synthetic_subset_accuracy": synthetic_acc,
            "gap": synthetic_acc - real_acc,
        }
    return results


def _threshold_series(columns: list[str], thresholds: dict[str, float] | float) -> pd.Series:
    if isinstance(thresholds, dict):
        return pd.Series({column: thresholds.get(column, 0.5) for column in columns})
    return pd.Series({column: float(thresholds) for column in columns})


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
