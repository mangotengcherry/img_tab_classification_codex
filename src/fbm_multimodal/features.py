from __future__ import annotations

import pandas as pd

from fbm_multimodal.measurement import MeasurementMap


def build_tabular_feature_frame(manifest: pd.DataFrame, measurement_map: MeasurementMap) -> pd.DataFrame:
    """Build metadata-aware tabular features without treating MSR suffix order as physical order."""
    raw_features = sorted(
        feature for feature in measurement_map.frame["feature_name"].tolist() if feature in manifest.columns
    )
    raw_frame = manifest[raw_features].astype(float).copy()
    raw_frame.columns = [f"raw::{column}" for column in raw_frame.columns]

    aggregate_rows = [measurement_map.aggregate_row(row) for _, row in manifest.iterrows()]
    aggregate_frame = pd.DataFrame(aggregate_rows, index=manifest.index).fillna(0.0)
    aggregate_frame = aggregate_frame.reindex(sorted(aggregate_frame.columns), axis=1)
    return pd.concat([raw_frame.reset_index(drop=True), aggregate_frame.reset_index(drop=True)], axis=1)


def build_late_fusion_frame(image_prob: pd.DataFrame, tabular_prob: pd.DataFrame) -> pd.DataFrame:
    """Create a late-fusion calibration matrix from unimodal probabilities."""
    labels = list(image_prob.columns)
    tabular = tabular_prob[labels]
    pieces = []

    image_prefixed = image_prob[labels].copy()
    image_prefixed.columns = [f"image_prob_{label}" for label in labels]
    pieces.append(image_prefixed)

    tabular_prefixed = tabular.copy()
    tabular_prefixed.columns = [f"tabular_prob_{label}" for label in labels]
    pieces.append(tabular_prefixed)

    diff = (image_prob[labels] - tabular).abs().round(12)
    diff.columns = [f"abs_diff_{label}" for label in labels]
    pieces.append(diff)
    return pd.concat(pieces, axis=1)
