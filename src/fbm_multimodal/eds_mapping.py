"""EDS test-item to wordline mapping helpers.

This module keeps the human-authored mapping table small:

``feature_name, eds_step, eds_item, wordline_position``

and derives model-routing flags for downstream pipelines. Features with a
wordline position can become WL residual measurements. Numeric EDS features are
CatBoost candidates unless the mapping explicitly disables them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fbm_multimodal.wl_residual_map import parse_wordline


MINIMAL_MAPPING_COLUMNS = {"feature_name", "eds_step", "eds_item", "wordline_position"}
METADATA_COLUMNS = {
    "sample_id",
    "chip_id",
    "row_idx",
    "split",
    "eval_group",
    "sample_type",
    "is_synthetic",
    "lot_id",
    "wafer_id",
}


def validate_eds_wordline_map(mapping: pd.DataFrame, eds_columns: list[str]) -> pd.DataFrame:
    """Validate and normalize a human-authored EDS-to-WL mapping table."""
    frame = mapping.copy()
    frame = _normalize_column_aliases(frame)
    missing = sorted(MINIMAL_MAPPING_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"EDS mapping is missing required columns: {missing}")

    frame["feature_name"] = frame["feature_name"].astype(str)
    if frame["feature_name"].duplicated().any():
        duplicates = frame.loc[frame["feature_name"].duplicated(), "feature_name"].tolist()
        raise ValueError(f"EDS mapping contains duplicate feature_name values: {duplicates}")

    eds_column_set = set(eds_columns)
    unknown = [feature for feature in frame["feature_name"] if feature not in eds_column_set]
    if unknown:
        raise ValueError(f"EDS mapping references feature columns not present in EDS data: {unknown}")

    frame["test_method"] = frame["eds_step"].astype(str)
    frame["test_item"] = frame["eds_item"].astype(str)
    frame["wordline"] = frame["wordline_position"].map(_parse_optional_wordline)
    frame["value_direction"] = frame.get("value_direction", "high_bad")
    frame["value_direction"] = frame["value_direction"].fillna("high_bad").astype(str).str.lower()
    invalid_direction = sorted(set(frame["value_direction"]) - {"high_bad", "low_bad"})
    if invalid_direction:
        raise ValueError(f"value_direction must be high_bad or low_bad, got: {invalid_direction}")

    if "include_in_wl_map" not in frame.columns:
        frame["include_in_wl_map"] = frame["wordline"].notna().astype(int)
    else:
        frame["include_in_wl_map"] = frame["include_in_wl_map"].map(_flag_to_int)
    if "include_in_catboost" not in frame.columns:
        frame["include_in_catboost"] = 1
    else:
        frame["include_in_catboost"] = frame["include_in_catboost"].map(_flag_to_int)

    missing_wl = frame["include_in_wl_map"].astype(bool) & frame["wordline"].isna()
    if missing_wl.any():
        bad = frame.loc[missing_wl, "feature_name"].tolist()
        raise ValueError(f"include_in_wl_map=1 requires wordline_position for: {bad}")

    ordered = [
        "feature_name",
        "eds_step",
        "eds_item",
        "test_method",
        "test_item",
        "wordline_position",
        "wordline",
        "value_direction",
        "include_in_wl_map",
        "include_in_catboost",
    ]
    extra = [column for column in frame.columns if column not in ordered]
    return frame[ordered + extra].reset_index(drop=True)


def wide_eds_to_wl_measurements(
    eds_tabular: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    sample_id_column: str = "sample_id",
    split_column: str = "split",
    eval_group_column: str = "eval_group",
    synthetic_column: str = "is_synthetic",
) -> pd.DataFrame:
    """Convert wide EDS tabular columns to long-form WL measurements."""
    if sample_id_column not in eds_tabular.columns:
        raise ValueError(f"EDS tabular is missing {sample_id_column!r}")
    normalized = validate_eds_wordline_map(mapping, eds_tabular.columns.tolist())
    wl_rows = normalized[normalized["include_in_wl_map"].astype(bool)]

    records: list[dict[str, object]] = []
    metadata_columns = [sample_id_column, split_column, eval_group_column, synthetic_column]
    for _, eds_row in eds_tabular.iterrows():
        for map_row in wl_rows.to_dict("records"):
            feature_name = map_row["feature_name"]
            value = eds_row[feature_name]
            if pd.isna(value):
                continue
            numeric = float(value)
            if map_row["value_direction"] == "low_bad":
                numeric = -numeric
            records.append(
                {
                    "sample_id": str(eds_row[sample_id_column]),
                    "split": str(eds_row[split_column]) if split_column in eds_tabular.columns else "",
                    "eval_group": str(eds_row[eval_group_column]) if eval_group_column in eds_tabular.columns else "",
                    "test_method": map_row["test_method"],
                    "wordline": int(map_row["wordline"]),
                    "value": numeric,
                    "feature_name": feature_name,
                    "test_item": map_row["test_item"],
                    "is_synthetic": bool(eds_row[synthetic_column]) if synthetic_column in eds_tabular.columns else False,
                }
            )
    columns = [
        "sample_id",
        "split",
        "eval_group",
        "test_method",
        "wordline",
        "value",
        "feature_name",
        "test_item",
        "is_synthetic",
    ]
    return pd.DataFrame(records, columns=columns)


def catboost_feature_columns(
    eds_tabular: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    label_columns: list[str] | None = None,
) -> list[str]:
    """Return mapped EDS columns enabled for CatBoost scalar training."""
    labels = set(label_columns or [])
    normalized = validate_eds_wordline_map(mapping, eds_tabular.columns.tolist())
    candidates = normalized.loc[normalized["include_in_catboost"].astype(bool), "feature_name"].tolist()
    excluded = METADATA_COLUMNS | labels
    if eds_tabular.empty:
        numeric_columns = set(eds_tabular.columns)
    else:
        numeric_columns = set(eds_tabular.select_dtypes(include=[np.number]).columns)
    return [column for column in candidates if column in numeric_columns and column not in excluded]


def read_eds_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    return pd.read_csv(source)


def write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".parquet":
        try:
            frame.to_parquet(target, index=False)
            return target
        except (ImportError, ValueError):
            fallback = target.with_suffix(".csv")
            frame.to_csv(fallback, index=False)
            return fallback
    frame.to_csv(target, index=False)
    return target


def _normalize_column_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "test_method": "eds_step",
        "test_item": "eds_item",
        "wordline": "wordline_position",
    }
    out = frame.copy()
    for old, new in aliases.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]
    return out


def _parse_optional_wordline(value: object) -> float:
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    return float(parse_wordline(value))


def _flag_to_int(value: object) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes", "y"})
    return int(bool(value))
