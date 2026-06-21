from __future__ import annotations

from dataclasses import dataclass

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


def evaluate_conditions(
    predictions: pd.DataFrame,
    *,
    labels: list[str],
    threshold: float = 0.5,
    condition_column: str = "condition",
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
    for condition, condition_frame in predictions.groupby(condition_column, sort=True):
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


def summarize_condition_report(summary: pd.DataFrame, targets: TargetConfig | None = None) -> dict[str, object]:
    """Create a compact JSON-serializable report header for condition evaluation."""
    if targets is None:
        targets = TargetConfig()
    if summary.empty:
        best_condition = None
    else:
        best_condition = str(summary.sort_values("kpi_product", ascending=False).iloc[0]["condition"])
    return {
        "best_condition_by_kpi": best_condition,
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


def _validate_prediction_frame(
    frame: pd.DataFrame,
    labels: list[str],
    condition_column: str,
    group_column: str,
) -> None:
    missing = [condition_column, group_column]
    for label in labels:
        missing.extend([f"true_{label}", f"prob_{label}"])
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
    pred = (subset[[f"prob_{label}" for label in labels]] >= threshold).astype(int).to_numpy()
    return float(np.mean(np.all(true == pred, axis=1)))


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
