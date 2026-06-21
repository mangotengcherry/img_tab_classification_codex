from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fbm_multimodal.active_learning import rank_unlabeled_for_review
from fbm_multimodal.condition_eval import (
    TargetConfig,
    aggregate_condition_runs,
    evaluate_condition_class_pair_metrics,
    evaluate_condition_per_class_metrics,
    evaluate_conditions,
    evaluate_threshold_grid,
    load_condition_predictions,
    render_condition_report,
    summarize_condition_report,
)
from fbm_multimodal.measurement import MeasurementMap


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate-map":
        return _validate_map(args)
    if args.command == "rank-unlabeled":
        return _rank_unlabeled(args)
    if args.command == "evaluate-conditions":
        return _evaluate_conditions(args)
    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fbm-mm",
        description="FBM image + electrical measurement multi-modal experiment utilities.",
    )
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser("validate-map", help="Report measurement_map.csv feature coverage.")
    validate.add_argument("--manifest", required=True, help="Chip manifest CSV containing MSR_* columns.")
    validate.add_argument("--measurement-map", required=True, help="Measurement mapping CSV.")

    rank = subparsers.add_parser("rank-unlabeled", help="Rank unlabeled chips for engineer review.")
    rank.add_argument("--candidates", required=True, help="Candidate predictions CSV.")
    rank.add_argument("--labels", required=True, help="Comma-separated label names.")
    rank.add_argument("--target-labels", required=True, help="Comma-separated target labels.")
    rank.add_argument("--budget", required=True, type=int, help="Number of chips to select.")
    rank.add_argument("--output", required=True, help="Output ranked CSV path.")
    rank.add_argument("--embedding-columns", default="", help="Optional comma-separated embedding columns.")

    evaluate = subparsers.add_parser(
        "evaluate-conditions",
        help="Evaluate condition-level single/composite subset accuracy and KPI gates.",
    )
    evaluate.add_argument("--predictions", default="", help="CSV with condition, eval_group, true_*, prob_* columns.")
    evaluate.add_argument(
        "--prediction-glob",
        default="",
        help="Glob for one CSV per condition. Files without condition column use the file stem.",
    )
    evaluate.add_argument("--labels", required=True, help="Comma-separated label names.")
    evaluate.add_argument("--output", required=True, help="Output condition summary CSV path.")
    evaluate.add_argument("--report-output", default="", help="Optional Markdown PASS/FAIL report output path.")
    evaluate.add_argument("--per-class-output", default="", help="Optional per-class precision/recall/F1 CSV path.")
    evaluate.add_argument("--class-pair-output", default="", help="Optional class-pair subset accuracy CSV path.")
    evaluate.add_argument("--run-column", default="", help="Optional seed/run column for per-run condition summaries.")
    evaluate.add_argument(
        "--aggregate-output",
        default="",
        help="Optional output CSV for condition-level aggregate metrics across runs.",
    )
    evaluate.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary labels.")
    evaluate.add_argument(
        "--threshold-grid",
        default="",
        help="Optional comma-separated thresholds. When set, the best threshold per condition is selected.",
    )
    evaluate.add_argument("--single-target", type=float, default=0.8, help="Target single-defect subset accuracy.")
    evaluate.add_argument("--composite-target", type=float, default=0.6, help="Target composite-defect subset accuracy.")
    evaluate.add_argument("--kpi-target", type=float, default=0.65, help="Target single*composite KPI product.")
    evaluate.add_argument(
        "--fail-on-miss",
        action="store_true",
        help="Return exit code 2 when no evaluated condition meets all targets.",
    )
    evaluate.add_argument(
        "--require-all-runs",
        action="store_true",
        help="With --fail-on-miss and --run-column, require a condition to pass every run.",
    )
    return parser


def _validate_map(args: argparse.Namespace) -> int:
    manifest = pd.read_csv(args.manifest)
    mapping = MeasurementMap.from_csv(args.measurement_map)
    msr_columns = [column for column in manifest.columns if column.startswith("MSR_")]
    coverage = mapping.coverage(msr_columns)
    payload = {
        "total_features": coverage.total_features,
        "mapped_features": coverage.mapped_features,
        "coverage_ratio": coverage.coverage_ratio,
        "missing_features": coverage.missing_features,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _rank_unlabeled(args: argparse.Namespace) -> int:
    candidates = pd.read_csv(args.candidates)
    labels = _split_csv_arg(args.labels)
    target_labels = _split_csv_arg(args.target_labels)
    embedding_columns = _split_csv_arg(args.embedding_columns)
    ranked = rank_unlabeled_for_review(
        candidates,
        label_columns=labels,
        target_labels=target_labels,
        budget=args.budget,
        embedding_columns=embedding_columns,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_path, index=False)
    return 0


def _evaluate_conditions(args: argparse.Namespace) -> int:
    predictions = _load_prediction_input(args.predictions, args.prediction_glob)
    labels = _split_csv_arg(args.labels)
    targets = TargetConfig(
        single_subset_accuracy=args.single_target,
        composite_subset_accuracy=args.composite_target,
        kpi_product=args.kpi_target,
    )
    threshold_grid = _split_float_arg(args.threshold_grid)
    run_column = args.run_column or None
    if args.require_all_runs and not run_column:
        raise ValueError("--require-all-runs requires --run-column")
    if threshold_grid:
        summary = evaluate_threshold_grid(
            predictions,
            labels=labels,
            thresholds=threshold_grid,
            run_column=run_column,
            targets=targets,
        )
    else:
        summary = evaluate_conditions(
            predictions,
            labels=labels,
            threshold=args.threshold,
            run_column=run_column,
            targets=targets,
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    aggregate = None
    if args.aggregate_output or args.require_all_runs:
        if not run_column:
            raise ValueError("--aggregate-output requires --run-column")
        aggregate = aggregate_condition_runs(summary, run_column=run_column)
    if args.aggregate_output:
        aggregate_path = Path(args.aggregate_output)
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate.to_csv(aggregate_path, index=False)
    if args.per_class_output:
        per_class = evaluate_condition_per_class_metrics(
            predictions,
            labels=labels,
            summary=summary,
            run_column=run_column,
        )
        per_class_path = Path(args.per_class_output)
        per_class_path.parent.mkdir(parents=True, exist_ok=True)
        per_class.to_csv(per_class_path, index=False)
    if args.class_pair_output:
        class_pair = evaluate_condition_class_pair_metrics(
            predictions,
            labels=labels,
            summary=summary,
            run_column=run_column,
        )
        class_pair_path = Path(args.class_pair_output)
        class_pair_path.parent.mkdir(parents=True, exist_ok=True)
        class_pair.to_csv(class_pair_path, index=False)
    if args.report_output:
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_condition_report(summary, aggregate=aggregate, targets=targets))
    print(json.dumps(summarize_condition_report(summary, targets), ensure_ascii=False, indent=2))
    if args.fail_on_miss and not _condition_gate_passed(
        summary,
        aggregate=aggregate,
        require_all_runs=args.require_all_runs,
    ):
        return 2
    return 0


def _load_prediction_input(predictions_path: str, prediction_glob: str) -> pd.DataFrame:
    if predictions_path and prediction_glob:
        raise ValueError("use only one of --predictions or --prediction-glob")
    if prediction_glob:
        return load_condition_predictions(prediction_glob)
    if predictions_path:
        return pd.read_csv(predictions_path)
    raise ValueError("one of --predictions or --prediction-glob is required")


def _split_csv_arg(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _split_float_arg(value: str) -> list[float]:
    return [float(part) for part in _split_csv_arg(value)]


def _condition_gate_passed(
    summary: pd.DataFrame,
    *,
    aggregate: pd.DataFrame | None,
    require_all_runs: bool,
) -> bool:
    if require_all_runs:
        if aggregate is None:
            raise ValueError("--require-all-runs requires --run-column")
        if aggregate.empty:
            return False
        return bool(aggregate["all_runs_meet_targets"].astype(bool).any())
    if summary.empty:
        return False
    return bool(summary["meets_all_targets"].astype(bool).any())


if __name__ == "__main__":
    raise SystemExit(main())
