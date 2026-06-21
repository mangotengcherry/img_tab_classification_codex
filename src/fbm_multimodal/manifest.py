from __future__ import annotations

import pandas as pd


METADATA_COLUMNS = {
    "chip_id",
    "image_path",
    "label_vector",
    "is_real",
    "is_synthetic",
    "is_pseudo_labeled",
    "label_cardinality",
    "wafer_position",
    "split",
}
EXCLUDED_PREFIXES = (
    "MSR_",
    "prob_",
    "image_prob_",
    "tabular_prob_",
    "fusion_prob_",
    "threshold_",
    "uncertainty_",
)


def label_columns(frame: pd.DataFrame) -> list[str]:
    """Infer multi-label target columns from a chip manifest frame."""
    labels: list[str] = []
    for column in frame.columns:
        if column in METADATA_COLUMNS:
            continue
        if any(column.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            continue
        labels.append(column)
    return labels


def assert_no_synthetic_or_pseudo_in_real_eval(frame: pd.DataFrame) -> list[str]:
    """Return chip IDs that violate real evaluation purity."""
    if "chip_id" not in frame.columns:
        raise ValueError("manifest must include chip_id")

    is_synthetic = _boolean_column(frame, "is_synthetic")
    is_pseudo = _boolean_column(frame, "is_pseudo_labeled")
    bad_rows = frame[is_synthetic | is_pseudo]
    return bad_rows["chip_id"].astype(str).tolist()


def _boolean_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column].fillna(False)
    if values.dtype == object:
        return values.astype(str).str.lower().isin({"1", "true", "yes", "y"})
    return values.astype(bool)
