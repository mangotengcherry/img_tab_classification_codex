from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetConfig:
    single_subset_accuracy: float = 0.8
    composite_subset_accuracy: float = 0.6
    kpi_product: float = 0.65

    @property
    def minimum_product_from_individual_targets(self) -> float:
        return self.single_subset_accuracy * self.composite_subset_accuracy

    @property
    def has_target_tension(self) -> bool:
        return self.minimum_product_from_individual_targets < self.kpi_product


def load_condition_predictions(path_pattern: str) -> pd.DataFrame:
    """Load one or more condition prediction CSV files.

    Files without a condition column inherit the file stem as their condition.
    Files with a condition column keep their explicit values.
    """
    paths = sorted(glob(path_pattern))
    if not paths:
        raise FileNotFoundError(f"no prediction files matched pattern: {path_pattern}")

    frames = []
    for path_text in paths:
        path = Path(path_text)
        frame = pd.read_csv(path)
        if "condition" not in frame.columns:
            frame.insert(0, "condition", path.stem)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def evaluate_conditions(
    predictions: pd.DataFrame,
    *,
    labels: list[str],
    threshold: float = 0.5,
    condition_column: str = "condition",
    run_column: str | None = None,
    group_column: str = "eval_group",
    single_group: str = "real_single",
    composite_group: str = "real_composite",
    synthetic_composite_group: str = "synthetic_composite",
    targets: TargetConfig | None = None,
) -> pd.DataFrame:
    """Evaluate condition-level subset accuracy and KPI gates from prediction rows."""
    if targets is None:
        targets = TargetConfig()
    _validate_prediction_frame(predictions, labels, condition_column, group_column)

    summaries = []
    group_keys = [condition_column]
    if run_column:
        if run_column not in predictions.columns:
            raise ValueError(f"run column not found: {run_column}")
        group_keys.append(run_column)

    groupby_arg: str | list[str] = group_keys[0] if len(group_keys) == 1 else group_keys
    for group_key, condition_frame in predictions.groupby(groupby_arg, sort=True):
        if run_column:
            condition = group_key[0]
            run_value = group_key[1]
        else:
            condition = group_key
            run_value = None
        single_acc = _subset_accuracy_for_group(condition_frame, labels, single_group, group_column, threshold)
        composite_acc = _subset_accuracy_for_group(condition_frame, labels, composite_group, group_column, threshold)
        synthetic_acc = _subset_accuracy_for_group(
            condition_frame,
            labels,
            synthetic_composite_group,
            group_column,
            threshold,
        )
        kpi = _safe_product(single_acc, composite_acc)
        summaries.append(
            {
                "condition": condition,
                **({run_column: run_value} if run_column else {}),
                "threshold": threshold,
                "single_subset_accuracy": single_acc,
                "composite_subset_accuracy": composite_acc,
                "synthetic_composite_subset_accuracy": synthetic_acc,
                "real_synthetic_composite_gap": _gap(synthetic_acc, composite_acc),
                "kpi_product": kpi,
                "meets_single_target": _meets(single_acc, targets.single_subset_accuracy),
                "meets_composite_target": _meets(composite_acc, targets.composite_subset_accuracy),
                "meets_kpi_target": _meets(kpi, targets.kpi_product),
                "meets_all_targets": (
                    _meets(single_acc, targets.single_subset_accuracy)
                    and _meets(composite_acc, targets.composite_subset_accuracy)
                    and _meets(kpi, targets.kpi_product)
                ),
                "required_composite_for_kpi_at_single": _required_other_metric(targets.kpi_product, single_acc),
                "required_single_for_kpi_at_composite": _required_other_metric(targets.kpi_product, composite_acc),
                "single_support": _support(condition_frame, single_group, group_column),
                "composite_support": _support(condition_frame, composite_group, group_column),
                "synthetic_composite_support": _support(condition_frame, synthetic_composite_group, group_column),
                "target_minima_product": targets.minimum_product_from_individual_targets,
                "target_tension": targets.has_target_tension,
            }
        )

    result = pd.DataFrame(summaries)
    if result.empty:
        return result
    return result.sort_values(["meets_all_targets", "kpi_product", "condition"], ascending=[False, False, True])


def aggregate_condition_runs(
    per_run_summary: pd.DataFrame,
    *,
    run_column: str,
) -> pd.DataFrame:
    """Aggregate per-run condition summaries into condition-level robustness metrics."""
    if run_column not in per_run_summary.columns:
        raise ValueError(f"run column not found: {run_column}")

    rows = []
    for condition, frame in per_run_summary.groupby("condition", sort=True):
        row = {
            "condition": condition,
            "num_runs": int(frame[run_column].nunique()),
            "all_runs_meet_targets": bool(frame["meets_all_targets"].all()),
            "any_run_meets_targets": bool(frame["meets_all_targets"].any()),
            "best_threshold_by_mean_kpi": _mode_or_first(frame["threshold"]),
        }
        for metric in ["single_subset_accuracy", "composite_subset_accuracy", "kpi_product"]:
            values = frame[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["all_runs_meet_targets", "kpi_product_mean", "condition"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def evaluate_threshold_grid(
    predictions: pd.DataFrame,
    *,
    labels: list[str],
    thresholds: list[float],
    condition_column: str = "condition",
    run_column: str | None = None,
    group_column: str = "eval_group",
    single_group: str = "real_single",
    composite_group: str = "real_composite",
    synthetic_composite_group: str = "synthetic_composite",
    targets: TargetConfig | None = None,
) -> pd.DataFrame:
    """Evaluate a threshold grid and keep the best threshold for each condition."""
    if not thresholds:
        raise ValueError("thresholds must not be empty")

    summaries = []
    for threshold in thresholds:
        threshold_summary = evaluate_conditions(
            predictions,
            labels=labels,
            threshold=threshold,
            condition_column=condition_column,
            run_column=run_column,
            group_column=group_column,
            single_group=single_group,
            composite_group=composite_group,
            synthetic_composite_group=synthetic_composite_group,
            targets=targets,
        )
        summaries.append(threshold_summary)

    all_results = pd.concat(summaries, ignore_index=True)
    ranked = all_results.sort_values(
        [
            "condition",
            *([run_column] if run_column else []),
            "meets_all_targets",
            "kpi_product",
            "single_subset_accuracy",
            "composite_subset_accuracy",
            "threshold",
        ],
        ascending=[True, False, False, False, False, True],
    )
    dedupe_columns = ["condition"] + ([run_column] if run_column else [])
    best_per_condition = ranked.drop_duplicates(subset=dedupe_columns, keep="first")
    return best_per_condition.sort_values(
        ["meets_all_targets", "kpi_product", "condition"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def summarize_condition_report(summary: pd.DataFrame, targets: TargetConfig | None = None) -> dict[str, object]:
    """Create a compact JSON-serializable report header for condition evaluation."""
    if targets is None:
        targets = TargetConfig()
    if summary.empty:
        best_condition = None
        best_threshold = None
    else:
        best_row = summary.sort_values("kpi_product", ascending=False).iloc[0]
        best_condition = str(best_row["condition"])
        best_threshold = float(best_row["threshold"]) if "threshold" in best_row.index else None
    return {
        "best_condition_by_kpi": best_condition,
        "best_threshold_by_kpi": best_threshold,
        "single_target": targets.single_subset_accuracy,
        "composite_target": targets.composite_subset_accuracy,
        "kpi_target": targets.kpi_product,
        "target_minima_product": targets.minimum_product_from_individual_targets,
        "target_tension": targets.has_target_tension,
        "required_composite_if_single_is_target": _required_other_metric(
            targets.kpi_product,
            targets.single_subset_accuracy,
        ),
        "required_single_if_composite_is_target": _required_other_metric(
            targets.kpi_product,
            targets.composite_subset_accuracy,
        ),
        "num_conditions": int(len(summary)),
        "num_conditions_meeting_all_targets": int(summary["meets_all_targets"].sum()) if not summary.empty else 0,
    }


def render_condition_report(
    summary: pd.DataFrame,
    *,
    aggregate: pd.DataFrame | None = None,
    targets: TargetConfig | None = None,
) -> str:
    """Render a compact Markdown PASS/FAIL report for condition evaluation."""
    if targets is None:
        targets = TargetConfig()
    lines = [
        "# FBM Condition Evaluation Report",
        "",
        f"- Single Target: {_fmt(targets.single_subset_accuracy)}",
        f"- Composite Target: {_fmt(targets.composite_subset_accuracy)}",
        f"- KPI Product Target: {_fmt(targets.kpi_product)}",
        f"- Minimum Product From Individual Targets: {_fmt(targets.minimum_product_from_individual_targets)}",
        "",
    ]

    if summary.empty:
        lines.extend(["Overall Status: FAIL", "", "No conditions were evaluated."])
        return "\n".join(lines) + "\n"

    ranked = summary.sort_values(["meets_all_targets", "kpi_product"], ascending=[False, False])
    best = ranked.iloc[0]
    passing = bool(best["meets_all_targets"])
    lines.append(f"Overall Status: {'PASS' if passing else 'FAIL'}")
    label = "Recommended Condition" if passing else "Best Available Condition"
    lines.append(f"{label}: {best['condition']}")
    lines.append(f"Threshold: {_fmt(best['threshold'])}")
    lines.append(f"Single Subset Accuracy: {_fmt(best['single_subset_accuracy'])}")
    lines.append(f"Composite Subset Accuracy: {_fmt(best['composite_subset_accuracy'])}")
    lines.append(f"KPI Product: {_fmt(best['kpi_product'])}")
    if not passing:
        lines.append(f"KPI Gap: {_fmt(max(0.0, targets.kpi_product - float(best['kpi_product'])))}")
        lines.append(f"Composite Needed At Observed Single: {_fmt(best['required_composite_for_kpi_at_single'])}")
        lines.append(f"Single Needed At Observed Composite: {_fmt(best['required_single_for_kpi_at_composite'])}")

    lines.extend(["", "## Condition Summary", ""])
    lines.append("| condition | threshold | single | composite | kpi | pass |")
    lines.append("| --- | ---: | ---: | ---: | ---: | :---: |")
    for _, row in ranked.iterrows():
        lines.append(
            "| "
            f"{row['condition']} | "
            f"{_fmt(row['threshold'])} | "
            f"{_fmt(row['single_subset_accuracy'])} | "
            f"{_fmt(row['composite_subset_accuracy'])} | "
            f"{_fmt(row['kpi_product'])} | "
            f"{'Y' if bool(row['meets_all_targets']) else 'N'} |"
        )

    if aggregate is not None and not aggregate.empty:
        lines.extend(["", "## Run Aggregate Summary", ""])
        lines.append("| condition | runs | mean single | min composite | mean kpi | all runs pass |")
        lines.append("| --- | ---: | ---: | ---: | ---: | :---: |")
        for _, row in aggregate.iterrows():
            lines.append(
                "| "
                f"{row['condition']} | "
                f"{int(row['num_runs'])} | "
                f"{_fmt(row['single_subset_accuracy_mean'])} | "
                f"{_fmt(row['composite_subset_accuracy_min'])} | "
                f"{_fmt(row['kpi_product_mean'])} | "
                f"{'Y' if bool(row['all_runs_meet_targets']) else 'N'} |"
            )

    return "\n".join(lines) + "\n"


def _validate_prediction_frame(
    frame: pd.DataFrame,
    labels: list[str],
    condition_column: str,
    group_column: str,
) -> None:
    missing = [condition_column, group_column]
    for label in labels:
        missing.append(f"true_{label}")
        if f"prob_{label}" not in frame.columns and f"pred_{label}" not in frame.columns:
            missing.append(f"prob_{label} or pred_{label}")
    missing = [column for column in missing if column not in frame.columns]
    if missing:
        raise ValueError(f"prediction frame is missing required columns: {missing}")


def _subset_accuracy_for_group(
    frame: pd.DataFrame,
    labels: list[str],
    group_name: str,
    group_column: str,
    threshold: float,
) -> float:
    subset = frame[frame[group_column] == group_name]
    if subset.empty:
        return float("nan")
    true = subset[[f"true_{label}" for label in labels]].astype(int).to_numpy()
    pred = _prediction_matrix(subset, labels, threshold)
    return float(np.mean(np.all(true == pred, axis=1)))


def _prediction_matrix(frame: pd.DataFrame, labels: list[str], threshold: float) -> np.ndarray:
    columns = []
    for label in labels:
        prob_column = f"prob_{label}"
        pred_column = f"pred_{label}"
        if prob_column in frame.columns:
            columns.append((frame[prob_column] >= threshold).astype(int))
        elif pred_column in frame.columns:
            columns.append(frame[pred_column].astype(int))
        else:
            raise ValueError(f"missing prediction column for label: {label}")
    return pd.concat(columns, axis=1).to_numpy()


def _support(frame: pd.DataFrame, group_name: str, group_column: str) -> int:
    return int((frame[group_column] == group_name).sum())


def _safe_product(left: float, right: float) -> float:
    if np.isnan(left) or np.isnan(right):
        return float("nan")
    return float(left * right)


def _gap(synthetic_acc: float, real_acc: float) -> float:
    if np.isnan(synthetic_acc) or np.isnan(real_acc):
        return float("nan")
    return float(synthetic_acc - real_acc)


def _meets(value: float, target: float) -> bool:
    return bool(not np.isnan(value) and value >= target)


def _required_other_metric(kpi_target: float, known_metric: float) -> float:
    if np.isnan(known_metric) or known_metric <= 0:
        return float("inf")
    return float(kpi_target / known_metric)


def _mode_or_first(values: pd.Series) -> float:
    modes = values.mode()
    if not modes.empty:
        return float(modes.iloc[0])
    return float(values.iloc[0])


def _fmt(value: object) -> str:
    numeric = float(value)
    if np.isinf(numeric):
        return "inf"
    if np.isnan(numeric):
        return "nan"
    return f"{numeric:.3f}"
