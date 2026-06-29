"""Load and validate real FBM tensor + EDS tabular inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fbm_multimodal.eds_mapping import read_eds_table


VALID_SPLITS = {"train", "valid", "test"}
REQUIRED_MANIFEST_COLUMNS = {"row_idx", "sample_id", "split", "eval_group"}


def load_fbm_tensor_dataset(fbm_dir: str | Path) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    """Load `fbm_images.npy`, `fbm_manifest.csv`, and `label_map.json`."""
    root = Path(fbm_dir)
    image_path = root / "fbm_images.npy"
    manifest_path = root / "fbm_manifest.csv"
    label_map_path = root / "label_map.json"
    if not image_path.exists():
        raise FileNotFoundError(f"missing FBM tensor file: {image_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing FBM manifest file: {manifest_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"missing label map file: {label_map_path}")

    images = np.load(image_path)
    manifest = pd.read_csv(manifest_path)
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    label_columns = list(label_map.get("label_columns", []))
    if not label_columns:
        raise ValueError("label_map.json must contain non-empty label_columns")
    _validate_manifest(manifest, label_columns=label_columns, n_images=images.shape[0], name="fbm_manifest")
    return images, manifest.reset_index(drop=True), label_map


def load_eds_tabular(path: str | Path) -> pd.DataFrame:
    """Load EDS tabular data from CSV or parquet and validate core columns."""
    frame = read_eds_table(path)
    _validate_sample_table(frame, name="eds_tabular")
    return frame.reset_index(drop=True)


def build_fusion_manifest(
    fbm_manifest: pd.DataFrame,
    eds_tabular: pd.DataFrame,
    *,
    label_columns: list[str],
) -> pd.DataFrame:
    """Join FBM and EDS availability into one modality-mask manifest."""
    _validate_manifest(fbm_manifest, label_columns=label_columns, n_images=None, name="fbm_manifest")
    _validate_sample_table(eds_tabular, name="eds_tabular")
    missing_labels = [column for column in label_columns if column not in eds_tabular.columns]
    if missing_labels:
        raise ValueError(f"eds_tabular is missing label columns: {missing_labels}")

    eds_labels = eds_tabular[["sample_id", *label_columns]].copy()
    joined = fbm_manifest.merge(eds_labels, on="sample_id", how="left", suffixes=("", "_eds"))
    has_eds = joined[[f"{column}_eds" for column in label_columns]].notna().any(axis=1)
    for column in label_columns:
        mismatch = has_eds & (joined[column].astype(float) != joined[f"{column}_eds"].astype(float))
        if mismatch.any():
            samples = joined.loc[mismatch, "sample_id"].astype(str).tolist()
            raise ValueError(f"label mismatch between FBM and EDS for {column}: {samples}")

    out = fbm_manifest.copy().reset_index(drop=True)
    out = out.rename(columns={"row_idx": "fbm_row_idx"})
    out["has_fbm_image"] = 1
    out["has_eds_tabular"] = has_eds.astype(int).to_numpy()
    out["has_wl_map"] = 0
    out["has_catboost_logits"] = 0
    return out


def _validate_manifest(
    frame: pd.DataFrame,
    *,
    label_columns: list[str],
    n_images: int | None,
    name: str,
) -> None:
    missing = sorted((REQUIRED_MANIFEST_COLUMNS | set(label_columns)) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    _validate_sample_table(frame, name=name)
    row_idx = pd.to_numeric(frame["row_idx"], errors="coerce")
    if row_idx.isna().any():
        raise ValueError(f"{name}.row_idx must be numeric")
    if n_images is not None:
        invalid = ~row_idx.astype(int).between(0, n_images - 1)
        if invalid.any():
            bad = frame.loc[invalid, "sample_id"].astype(str).tolist()
            raise ValueError(f"{name}.row_idx out of FBM tensor bounds for: {bad}")


def _validate_sample_table(frame: pd.DataFrame, *, name: str) -> None:
    missing = sorted({"sample_id", "split", "eval_group"} - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"].astype(str).tolist()
        raise ValueError(f"{name} contains duplicate sample_id values: {duplicates}")
    splits = set(frame["split"].astype(str))
    invalid_splits = sorted(splits - VALID_SPLITS)
    if invalid_splits:
        raise ValueError(f"{name}.split must be one of {sorted(VALID_SPLITS)}, got: {invalid_splits}")
