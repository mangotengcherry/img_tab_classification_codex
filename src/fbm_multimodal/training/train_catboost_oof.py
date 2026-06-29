"""Offline CatBoost one-vs-rest OOF logit generation.

CatBoost is an optional runtime dependency. Importing this module does not
require it; the default estimator factory raises a clear error only when real
training is requested without CatBoost installed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


EstimatorFactory = Callable[[int, int], object]


@dataclass
class CatBoostOOFResult:
    train_oof_logits: pd.DataFrame
    split_logits: dict[str, pd.DataFrame]
    models_by_class: dict[int, list[object]]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


def train_catboost_oof_logits(
    features: np.ndarray | pd.DataFrame,
    labels: np.ndarray | pd.DataFrame,
    *,
    sample_ids: list[str],
    split: list[str] | np.ndarray,
    is_synthetic: list[bool] | np.ndarray | None = None,
    num_folds: int = 5,
    random_seed: int = 42,
    output_dir: str | Path | None = None,
    model_factory: EstimatorFactory | None = None,
) -> CatBoostOOFResult:
    """Train one-vs-rest fold models and return train OOF plus eval logits."""
    x = _as_2d_float(features, "features")
    y = _as_2d_float(labels, "labels").astype(int)
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and labels must have the same row count")
    if len(sample_ids) != x.shape[0]:
        raise ValueError("sample_ids length must match feature rows")
    split_arr = np.asarray(split).astype(str)
    if split_arr.shape[0] != x.shape[0]:
        raise ValueError("split length must match feature rows")
    synthetic = np.zeros(x.shape[0], dtype=bool) if is_synthetic is None else np.asarray(is_synthetic, dtype=bool)
    if synthetic.shape[0] != x.shape[0]:
        raise ValueError("is_synthetic length must match feature rows")

    real_train_mask = (split_arr == "train") & (~synthetic)
    train_indices = np.flatnonzero(real_train_mask)
    if len(train_indices) < 2:
        raise ValueError("at least two real train samples are required for OOF logits")

    warnings_out: list[str] = []
    n_folds = min(int(num_folds), len(train_indices))
    if n_folds < num_folds:
        warnings_out.append(f"num_folds reduced from {num_folds} to {n_folds} because train support is small")
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    using_default_factory = model_factory is None
    if model_factory is None:
        model_factory = _default_catboost_factory
    warnings_out.append("using KFold fallback for multi-label OOF splitting")

    n_classes = y.shape[1]
    oof_logits = np.zeros((len(train_indices), n_classes), dtype=float)
    models_by_class: dict[int, list[object]] = {class_idx: [] for class_idx in range(n_classes)}

    train_x = x[train_indices]
    train_y = y[train_indices]
    for class_idx in range(n_classes):
        for fold_idx, (fold_train_pos, fold_valid_pos) in enumerate(splitter.split(train_x)):
            fold_seed = random_seed + class_idx * 1009 + fold_idx
            y_fold = train_y[fold_train_pos, class_idx]
            if using_default_factory and np.unique(y_fold).size < 2:
                model = _ConstantProbabilityEstimator(float(y_fold.mean()))
            else:
                model = model_factory(class_idx, fold_seed)
                model.fit(train_x[fold_train_pos], y_fold)
            models_by_class[class_idx].append(model)
            prob = _positive_probability(model, train_x[fold_valid_pos])
            oof_logits[fold_valid_pos, class_idx] = _logit(prob)

    train_oof = _logit_frame(
        [sample_ids[i] for i in train_indices],
        oof_logits,
    )

    split_logits: dict[str, pd.DataFrame] = {}
    for split_name in sorted(set(split_arr) - {"train"}):
        eval_indices = np.flatnonzero((split_arr == split_name) & (~synthetic))
        split_logits[split_name] = _predict_fold_ensemble_logits(
            x[eval_indices],
            [sample_ids[i] for i in eval_indices],
            models_by_class,
        )

    metadata: dict[str, object] = {
        "train_prediction_mode": "oof",
        "synthetic_excluded": True,
        "splitter": "KFold",
        "num_folds": int(n_folds),
        "random_seed": int(random_seed),
        "n_classes": int(n_classes),
        "train_real_sample_count": int(len(train_indices)),
        "eval_splits": sorted(split_logits),
    }
    if output_dir is not None:
        _write_outputs(Path(output_dir), train_oof, split_logits, models_by_class, metadata, warnings_out)
    metadata["warnings"] = list(warnings_out)

    return CatBoostOOFResult(
        train_oof_logits=train_oof,
        split_logits=split_logits,
        models_by_class=models_by_class,
        warnings=warnings_out,
        metadata=metadata,
    )


class _ConstantProbabilityEstimator:
    def __init__(self, probability: float) -> None:
        self.probability = float(np.clip(probability, 1.0e-6, 1.0 - 1.0e-6))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        positive = np.full(x.shape[0], self.probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])


def _default_catboost_factory(class_index: int, random_seed: int) -> object:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError(
            "CatBoost is required for default CatBoost OOF training. "
            "Install the optional dependency or pass model_factory for tests."
        ) from exc
    return CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=False,
    )


def _as_2d_float(values: np.ndarray | pd.DataFrame, name: str) -> np.ndarray:
    array = values.to_numpy(dtype=float) if isinstance(values, pd.DataFrame) else np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return array


def _positive_probability(model: object, x: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict_proba(x), dtype=float)
    if proba.ndim == 1:
        return proba
    if proba.shape[1] == 1:
        return proba[:, 0]
    return proba[:, 1]


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    return np.log(p / (1.0 - p))


def _logit_frame(sample_ids: list[str], logits: np.ndarray) -> pd.DataFrame:
    data: dict[str, object] = {"sample_id": sample_ids}
    for class_idx in range(logits.shape[1]):
        data[f"cat_logit_{class_idx}"] = logits[:, class_idx]
    data["has_catboost_logits"] = np.ones(logits.shape[0], dtype=float)
    return pd.DataFrame(data)


def _predict_fold_ensemble_logits(
    x: np.ndarray,
    sample_ids: list[str],
    models_by_class: dict[int, list[object]],
) -> pd.DataFrame:
    n_classes = len(models_by_class)
    logits = np.zeros((x.shape[0], n_classes), dtype=float)
    if x.shape[0] == 0:
        return _logit_frame(sample_ids, logits)
    for class_idx, models in models_by_class.items():
        fold_probs = [_positive_probability(model, x) for model in models]
        logits[:, class_idx] = _logit(np.mean(np.vstack(fold_probs), axis=0))
    return _logit_frame(sample_ids, logits)


def _write_outputs(
    output_dir: Path,
    train_oof: pd.DataFrame,
    split_logits: dict[str, pd.DataFrame],
    models_by_class: dict[int, list[object]],
    metadata: dict[str, object],
    warnings_out: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_table(train_oof, output_dir / "train_oof_logits.parquet", warnings_out)
    for split_name, frame in split_logits.items():
        _write_table(frame, output_dir / f"{split_name}_logits.parquet", warnings_out)
    _write_model_artifacts(models_by_class, output_dir / "models", warnings_out)
    metadata_with_warnings = {**metadata, "warnings": list(warnings_out)}
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata_with_warnings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "warnings.txt").write_text("\n".join(warnings_out), encoding="utf-8")


def _write_table(frame: pd.DataFrame, parquet_path: Path, warnings_out: list[str]) -> None:
    try:
        frame.to_parquet(parquet_path, index=False)
    except (ImportError, ValueError) as exc:
        csv_path = parquet_path.with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        message = f"could not write {parquet_path.name} as parquet ({exc}); wrote {csv_path.name} instead"
        warnings.warn(message)
        warnings_out.append(message)


def _write_model_artifacts(
    models_by_class: dict[int, list[object]],
    models_dir: Path,
    warnings_out: list[str],
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for class_idx, models in models_by_class.items():
        for fold_idx, model in enumerate(models):
            path = models_dir / f"class_{class_idx}_fold_{fold_idx}.pkl"
            try:
                with path.open("wb") as handle:
                    pickle.dump(model, handle)
            except Exception as exc:  # pragma: no cover - depends on estimator picklability
                message = f"could not pickle {path.name}: {exc}"
                warnings.warn(message)
                warnings_out.append(message)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one-vs-rest CatBoost OOF logits.")
    parser.add_argument("--features", required=True, help="CSV with sample_id, split, feature columns.")
    parser.add_argument("--labels", required=True, help="CSV with sample_id and one binary column per class.")
    parser.add_argument("--label-columns", required=True, help="Comma-separated label columns from --labels.")
    parser.add_argument("--output-dir", required=True, help="Output directory for logit tables.")
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--synthetic-column", default="is_synthetic")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args(argv)

    features = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels)
    merged = features.merge(labels, on=args.sample_id_column, how="inner", suffixes=("", "_label"))
    label_columns = [part.strip() for part in args.label_columns.split(",") if part.strip()]
    metadata_columns = {args.sample_id_column, args.split_column, args.synthetic_column}
    feature_columns = [column for column in features.columns if column not in metadata_columns]

    result = train_catboost_oof_logits(
        merged[feature_columns],
        merged[label_columns],
        sample_ids=merged[args.sample_id_column].astype(str).tolist(),
        split=merged[args.split_column].astype(str).tolist(),
        is_synthetic=(
            merged[args.synthetic_column].astype(bool).to_numpy()
            if args.synthetic_column in merged.columns
            else None
        ),
        num_folds=args.num_folds,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
    )
    for warning in result.warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
