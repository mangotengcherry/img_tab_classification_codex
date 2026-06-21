from __future__ import annotations

import numpy as np
import pandas as pd


def rank_unlabeled_for_review(
    candidates: pd.DataFrame,
    *,
    label_columns: list[str],
    target_labels: list[str],
    budget: int,
    embedding_columns: list[str] | None = None,
    high_confidence_threshold: float = 0.95,
    disagreement_threshold: float = 0.5,
    uncertainty_threshold: float = 0.8,
) -> pd.DataFrame:
    """Rank unlabeled chips for engineer review using practical acquisition signals."""
    if budget <= 0:
        return candidates.head(0).copy()

    scored = candidates.copy()
    scored["_target_confidence"] = _target_confidence(scored, target_labels)
    scored["_uncertainty"] = _uncertainty(scored, label_columns)
    scored["_disagreement"] = _image_tabular_disagreement(scored, label_columns)
    scored["_cluster_score"] = _cluster_representative_score(scored, embedding_columns)

    reasons = []
    priorities = []
    scores = []
    for row in scored.to_dict("records"):
        if row["_disagreement"] >= disagreement_threshold:
            reasons.append("image_tabular_disagreement")
            priorities.append(0)
            scores.append(row["_disagreement"])
        elif row["_target_confidence"] >= high_confidence_threshold:
            reasons.append("high_confidence_target")
            priorities.append(1)
            scores.append(row["_target_confidence"])
        elif row["_uncertainty"] >= uncertainty_threshold:
            reasons.append("high_uncertainty")
            priorities.append(2)
            scores.append(row["_uncertainty"])
        else:
            reasons.append("embedding_cluster_representative")
            priorities.append(3)
            scores.append(row["_cluster_score"])

    scored["selection_reason"] = reasons
    scored["_priority"] = priorities
    scored["_score"] = scores
    ranked = scored.sort_values(["_priority", "_score", "chip_id"], ascending=[True, False, True])
    return ranked.head(budget).drop(columns=[c for c in ranked.columns if c.startswith("_")])


def _target_confidence(frame: pd.DataFrame, target_labels: list[str]) -> pd.Series:
    probability_columns = [f"prob_{label}" for label in target_labels if f"prob_{label}" in frame.columns]
    if not probability_columns:
        return pd.Series(0.0, index=frame.index)
    return frame[probability_columns].max(axis=1)


def _uncertainty(frame: pd.DataFrame, label_columns: list[str]) -> pd.Series:
    probability_columns = [f"prob_{label}" for label in label_columns if f"prob_{label}" in frame.columns]
    if not probability_columns:
        return pd.Series(0.0, index=frame.index)
    uncertainty = 1.0 - 2.0 * (frame[probability_columns] - 0.5).abs()
    return uncertainty.clip(lower=0.0, upper=1.0).max(axis=1)


def _image_tabular_disagreement(frame: pd.DataFrame, label_columns: list[str]) -> pd.Series:
    disagreements = []
    for label in label_columns:
        image_column = f"image_prob_{label}"
        tabular_column = f"tabular_prob_{label}"
        if image_column in frame.columns and tabular_column in frame.columns:
            disagreements.append((frame[image_column] - frame[tabular_column]).abs())
    if not disagreements:
        return pd.Series(0.0, index=frame.index)
    return pd.concat(disagreements, axis=1).max(axis=1)


def _cluster_representative_score(frame: pd.DataFrame, embedding_columns: list[str] | None) -> pd.Series:
    if not embedding_columns:
        return pd.Series(0.0, index=frame.index)
    available = [column for column in embedding_columns if column in frame.columns]
    if not available:
        return pd.Series(0.0, index=frame.index)
    values = frame[available].astype(float).to_numpy()
    center = values.mean(axis=0)
    distance = np.linalg.norm(values - center, axis=1)
    if distance.max() == 0:
        return pd.Series(1.0, index=frame.index)
    score = 1.0 - distance / distance.max()
    return pd.Series(score, index=frame.index)
