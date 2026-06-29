"""WL raw measurement to high-side residual map tensors.

The tensorizer fits robust train-population baselines per
``(test_method, wl_bin)`` and transforms raw measurement rows into compact
``[channels, wl_bins, test_methods]`` tensors. It intentionally filters fitting
to train real rows when split/eval metadata is present so validation, test, and
synthetic rows cannot leak into the residual baseline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_WL_CHANNELS = [
    "mean_residual",
    "max_residual",
    "std_residual",
    "observed_mask",
    "count_ratio",
    "source_count_norm",
]

VALUE_CHANNELS = {"mean_residual", "max_residual", "std_residual"}


def parse_wordline(value: object) -> int:
    """Parse integer wordline values and strings like ``WL000``."""
    if pd.isna(value):
        raise ValueError("wordline value is missing")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    raise ValueError(f"could not parse wordline value: {value!r}")


@dataclass
class WLResidualMapTensorizer:
    """Fit train-only robust baselines and emit residual map tensors."""

    sample_id_column: str = "sample_id"
    test_method_column: str = "test_method"
    wl_column: str = "wordline"
    value_column: str = "value"
    split_column: str = "split"
    eval_group_column: str = "eval_group"
    synthetic_column: str = "is_synthetic"
    train_split_value: str = "train"
    wl_min: int = 0
    wl_max: int = 200
    num_wl_bins: int = 20
    test_methods: Iterable[str] | None = None
    channels: list[str] = field(default_factory=lambda: list(DEFAULT_WL_CHANNELS))
    eps: float = 1.0e-6
    clip_max: float = 10.0

    baseline_: pd.DataFrame | None = None
    expected_count_: dict[tuple[str, int], float] = field(default_factory=dict)
    test_methods_: list[str] = field(default_factory=list)
    fit_sample_ids_: set[str] = field(default_factory=set)

    @property
    def channel_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.channels)}

    @property
    def test_method_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.test_methods_)}

    def fit(self, train_measurement_df: pd.DataFrame) -> "WLResidualMapTensorizer":
        frame = self._prepare_frame(train_measurement_df)
        frame = self._filter_fit_frame(frame)
        if frame.empty:
            raise ValueError("no train real measurement rows available for WL residual baseline fitting")

        self.fit_sample_ids_ = set(frame[self.sample_id_column].astype(str).unique().tolist())
        if self.test_methods is None:
            self.test_methods_ = sorted(frame[self.test_method_column].astype(str).unique().tolist())
        else:
            self.test_methods_ = [str(method) for method in self.test_methods]

        grouped = frame.groupby([self.test_method_column, "wl_bin"], sort=True)[self.value_column]
        summary = grouped.agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        ).reset_index()
        summary["iqr_raw"] = summary["q75"] - summary["q25"]

        method_iqr = frame.groupby(self.test_method_column)[self.value_column].apply(_iqr).to_dict()
        global_iqr = _positive_or_none(_iqr(frame[self.value_column])) or 1.0
        scales: list[float] = []
        for row in summary.to_dict("records"):
            scale = _positive_or_none(row["iqr_raw"])
            if scale is None:
                scale = _positive_or_none(method_iqr.get(row[self.test_method_column]))
            if scale is None:
                scale = global_iqr
            scales.append(float(scale if scale > 0 else 1.0))
        summary["scale"] = scales
        self.baseline_ = summary[[self.test_method_column, "wl_bin", "median", "scale"]].copy()

        counts = (
            frame.groupby([self.sample_id_column, self.test_method_column, "wl_bin"], sort=True)
            .size()
            .rename("count")
            .reset_index()
        )
        expected = counts.groupby([self.test_method_column, "wl_bin"])["count"].median()
        self.expected_count_ = {
            (str(method), int(bin_idx)): float(count)
            for (method, bin_idx), count in expected.to_dict().items()
            if float(count) > 0
        }
        return self

    def transform(self, measurement_df_for_one_or_many_samples: pd.DataFrame) -> dict[str, np.ndarray]:
        if self.baseline_ is None:
            raise ValueError("WLResidualMapTensorizer must be fit before transform")
        frame = self._prepare_frame(measurement_df_for_one_or_many_samples)
        if frame.empty:
            return {}

        baseline = self.baseline_.rename(columns={"median": "_median", "scale": "_scale"})
        joined = frame.merge(baseline, on=[self.test_method_column, "wl_bin"], how="left")
        joined = joined[joined[self.test_method_column].isin(self.test_methods_)]
        joined = joined.dropna(subset=["_median", "_scale"])
        if joined.empty:
            return {
                str(sample_id): self._empty_tensor()
                for sample_id in frame[self.sample_id_column].astype(str).unique().tolist()
            }

        residual = (joined[self.value_column].to_numpy(dtype=float) - joined["_median"].to_numpy(dtype=float)) / (
            joined["_scale"].to_numpy(dtype=float) + self.eps
        )
        joined = joined.copy()
        joined["_residual"] = np.clip(np.maximum(0.0, residual), 0.0, self.clip_max)

        out: dict[str, np.ndarray] = {}
        for sample_id in frame[self.sample_id_column].astype(str).unique().tolist():
            out[sample_id] = self._empty_tensor()

        ch = self.channel_index
        grouped = joined.groupby([self.sample_id_column, self.test_method_column, "wl_bin"], sort=True)
        for (sample_id_raw, method_raw, bin_raw), cell in grouped:
            sample_id = str(sample_id_raw)
            method = str(method_raw)
            if sample_id not in out or method not in self.test_method_index:
                continue
            bin_idx = int(bin_raw)
            method_idx = self.test_method_index[method]
            values = cell["_residual"].to_numpy(dtype=float)
            tensor = out[sample_id]
            if "mean_residual" in ch:
                tensor[ch["mean_residual"], bin_idx, method_idx] = float(values.mean())
            if "max_residual" in ch:
                tensor[ch["max_residual"], bin_idx, method_idx] = float(values.max())
            if "std_residual" in ch:
                tensor[ch["std_residual"], bin_idx, method_idx] = float(values.std(ddof=0))
            if "observed_mask" in ch:
                tensor[ch["observed_mask"], bin_idx, method_idx] = 1.0
            if "count_ratio" in ch:
                expected = self.expected_count_.get((method, bin_idx), max(float(len(values)), 1.0))
                tensor[ch["count_ratio"], bin_idx, method_idx] = float(np.clip(len(values) / expected, 0.0, 1.0))
            if "source_count_norm" in ch:
                tensor[ch["source_count_norm"], bin_idx, method_idx] = 1.0
        return out

    def save(self, path: str | Path) -> None:
        if self.baseline_ is None:
            raise ValueError("cannot save an unfitted WLResidualMapTensorizer")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "sample_id_column": self.sample_id_column,
                "test_method_column": self.test_method_column,
                "wl_column": self.wl_column,
                "value_column": self.value_column,
                "split_column": self.split_column,
                "eval_group_column": self.eval_group_column,
                "synthetic_column": self.synthetic_column,
                "train_split_value": self.train_split_value,
                "wl_min": self.wl_min,
                "wl_max": self.wl_max,
                "num_wl_bins": self.num_wl_bins,
                "test_methods": self.test_methods_,
                "channels": self.channels,
                "eps": self.eps,
                "clip_max": self.clip_max,
            },
            "baseline": self.baseline_.to_dict("records"),
            "expected_count": [
                {"test_method": method, "wl_bin": bin_idx, "expected_count": count}
                for (method, bin_idx), count in self.expected_count_.items()
            ],
            "fit_sample_ids": sorted(self.fit_sample_ids_),
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_tensor_cache(self, tensors: dict[str, np.ndarray], path: str | Path) -> None:
        """Save transformed sample tensors to a compact cache file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        sample_ids = sorted(str(sample_id) for sample_id in tensors)
        if sample_ids:
            stacked = np.stack([np.asarray(tensors[sample_id], dtype=np.float32) for sample_id in sample_ids], axis=0)
        else:
            stacked = np.zeros((0, len(self.channels), self.num_wl_bins, len(self.test_methods_)), dtype=np.float32)
        np.savez_compressed(
            target,
            sample_ids=np.asarray(sample_ids, dtype=str),
            tensors=stacked,
        )

    @staticmethod
    def load_tensor_cache(path: str | Path) -> dict[str, np.ndarray]:
        """Load a tensor cache written by :meth:`save_tensor_cache`."""
        with np.load(Path(path), allow_pickle=False) as payload:
            sample_ids = [str(sample_id) for sample_id in payload["sample_ids"].tolist()]
            tensors = np.asarray(payload["tensors"], dtype=np.float32)
        return {sample_id: tensors[idx] for idx, sample_id in enumerate(sample_ids)}

    @classmethod
    def load(cls, path: str | Path) -> "WLResidualMapTensorizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config = payload["config"]
        tensorizer = cls(**config)
        tensorizer.test_methods_ = [str(method) for method in config["test_methods"]]
        tensorizer.baseline_ = pd.DataFrame(payload["baseline"])
        tensorizer.expected_count_ = {
            (str(row["test_method"]), int(row["wl_bin"])): float(row["expected_count"])
            for row in payload["expected_count"]
        }
        tensorizer.fit_sample_ids_ = set(payload.get("fit_sample_ids", []))
        return tensorizer

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {self.sample_id_column, self.test_method_column, self.wl_column, self.value_column}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"measurement dataframe is missing required columns: {missing}")
        prepared = frame.copy()
        prepared[self.sample_id_column] = prepared[self.sample_id_column].astype(str)
        prepared[self.test_method_column] = prepared[self.test_method_column].astype(str)
        prepared[self.value_column] = pd.to_numeric(prepared[self.value_column], errors="coerce")
        prepared = prepared.dropna(subset=[self.value_column])
        prepared["_wl_numeric"] = prepared[self.wl_column].map(parse_wordline)
        in_range = prepared["_wl_numeric"].between(self.wl_min, self.wl_max)
        prepared = prepared[in_range].copy()
        span = (self.wl_max - self.wl_min) + 1
        prepared["wl_bin"] = np.floor(
            ((prepared["_wl_numeric"].to_numpy(dtype=float) - self.wl_min) / span) * self.num_wl_bins
        ).astype(int)
        prepared["wl_bin"] = prepared["wl_bin"].clip(0, self.num_wl_bins - 1)
        return prepared

    def _filter_fit_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        mask = np.ones(len(frame), dtype=bool)
        if self.split_column in frame.columns:
            mask &= frame[self.split_column].astype(str).to_numpy() == self.train_split_value
        if self.synthetic_column in frame.columns:
            synthetic = frame[self.synthetic_column].astype(bool).to_numpy()
            mask &= ~synthetic
        if self.eval_group_column in frame.columns:
            eval_group = frame[self.eval_group_column].astype(str).str.lower()
            mask &= ~eval_group.str.startswith("synthetic").to_numpy()
        return frame[mask].copy()

    def _empty_tensor(self) -> np.ndarray:
        return np.zeros((len(self.channels), self.num_wl_bins, len(self.test_methods_)), dtype=np.float32)


def _iqr(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.quantile(0.75) - numeric.quantile(0.25))


def _positive_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    if numeric <= 0:
        return None
    return numeric
