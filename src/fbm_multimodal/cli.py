from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fbm_multimodal.active_learning import rank_unlabeled_for_review
from fbm_multimodal.condition_eval import TargetConfig, evaluate_conditions, summarize_condition_report
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
    evaluate.add_argument("--predictions", required=True, help="CSV with condition, eval_group, true_*, prob_* columns.")
    evaluate.add_argument("--labels", required=True, help="Comma-separated label names.")
    evaluate.add_argument("--output", required=True, help="Output condition summary CSV path.")
    evaluate.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary labels.")
    evaluate.add_argument("--single-target", type=float, default=0.8, help="Target single-defect subset accuracy.")
    evaluate.add_argument("--composite-target", type=float, default=0.6, help="Target composite-defect subset accuracy.")
    evaluate.add_argument("--kpi-target", type=float, default=0.65, help="Target single*composite KPI product.")
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
    predictions = pd.read_csv(args.predictions)
    labels = _split_csv_arg(args.labels)
    targets = TargetConfig(
        single_subset_accuracy=args.single_target,
        composite_subset_accuracy=args.composite_target,
        kpi_product=args.kpi_target,
    )
    summary = evaluate_conditions(
        predictions,
        labels=labels,
        threshold=args.threshold,
        targets=targets,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(json.dumps(summarize_condition_report(summary, targets), ensure_ascii=False, indent=2))
    return 0


def _split_csv_arg(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
