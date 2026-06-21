from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"feature_name", "measurement_condition", "measurement_type"}
OPTIONAL_COLUMNS = {"wl_index", "physical_region", "physical_order"}
VALID_REGIONS = {"top", "middle", "bottom", "unknown"}


@dataclass(frozen=True)
class MappingCoverage:
    total_features: int
    mapped_features: int
    missing_features: list[str]

    @property
    def coverage_ratio(self) -> float:
        if self.total_features == 0:
            return 1.0
        return self.mapped_features / self.total_features


@dataclass(frozen=True)
class MeasurementMap:
    """Metadata map from raw MSR feature names to measurement and physical context."""

    frame: pd.DataFrame

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "MeasurementMap":
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"measurement map is missing required columns: {missing}")

        normalized = frame.copy()
        for column in OPTIONAL_COLUMNS:
            if column not in normalized.columns:
                normalized[column] = np.nan

        normalized["feature_name"] = normalized["feature_name"].astype(str)
        normalized["measurement_condition"] = normalized["measurement_condition"].fillna("unknown").astype(str)
        normalized["measurement_type"] = normalized["measurement_type"].fillna("unknown").astype(str)
        normalized["physical_region"] = normalized["physical_region"].fillna("unknown").astype(str)
        normalized.loc[~normalized["physical_region"].isin(VALID_REGIONS), "physical_region"] = "unknown"
        return cls(normalized.reset_index(drop=True))

    @classmethod
    def from_csv(cls, path: str) -> "MeasurementMap":
        return cls.from_frame(pd.read_csv(path))

    def coverage(self, feature_columns: list[str]) -> MappingCoverage:
        mapped = set(self.frame["feature_name"])
        requested = list(feature_columns)
        missing = [feature for feature in requested if feature not in mapped]
        return MappingCoverage(
            total_features=len(requested),
            mapped_features=len(requested) - len(missing),
            missing_features=missing,
        )

    def features_by_physical_order(self) -> list[str]:
        ordered = self.frame.dropna(subset=["physical_order"]).copy()
        if ordered.empty:
            return []
        ordered["physical_order"] = pd.to_numeric(ordered["physical_order"], errors="coerce")
        ordered = ordered.dropna(subset=["physical_order"])
        ordered = ordered.sort_values(["physical_order", "feature_name"], kind="mergesort")
        return ordered["feature_name"].tolist()

    def aggregate_row(self, row: pd.Series) -> dict[str, float]:
        groups: dict[str, list[float]] = {}
        for record in self.frame.to_dict("records"):
            feature_name = record["feature_name"]
            if feature_name not in row.index:
                continue
            value = row[feature_name]
            if pd.isna(value):
                continue
            numeric = float(value)
            self._append_group(groups, f"region={record['physical_region']}", numeric)
            self._append_group(groups, f"condition={record['measurement_condition']}", numeric)
            self._append_group(groups, f"type={record['measurement_type']}", numeric)

        features: dict[str, float] = {}
        for group_name, values in groups.items():
            arr = np.asarray(values, dtype=np.float32)
            features[f"{group_name}__mean"] = float(arr.mean())
            features[f"{group_name}__max"] = float(arr.max())
            features[f"{group_name}__sum"] = float(arr.sum())
            features[f"{group_name}__nonzero_fraction"] = float(np.mean(arr > 0))
        return features

    @staticmethod
    def _append_group(groups: dict[str, list[float]], group_name: str, value: float) -> None:
        groups.setdefault(group_name, []).append(value)
