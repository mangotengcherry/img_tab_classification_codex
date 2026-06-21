"""Standalone CLI for fusion evaluation.

Run with:

    PYTHONPATH=src python3 -m fbm_multimodal.fusion \
        --predictions outputs/fusion_predictions.csv \
        --labels defect_a,defect_b,defect_c \
        --identity-labels defect_b \
        --output outputs/fusion_report.md

This is intentionally separate from ``fbm_multimodal.cli`` so it never collides
with concurrent edits to the core CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fbm_multimodal.fusion.fusion_eval import evaluate_fusion


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fbm-fusion-eval",
        description="Evaluate image/tabular/fusion heads with collapse + identity-slice diagnostics.",
    )
    parser.add_argument("--predictions", required=True, help="Predictions CSV (see docs/fusion_eval_quickstart.md).")
    parser.add_argument("--labels", required=True, help="Comma-separated label names.")
    parser.add_argument("--identity-labels", default="", help="Comma-separated electrical-only (identity) labels.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed threshold for all labels.")
    parser.add_argument("--group-column", default="eval_group")
    parser.add_argument("--output", default="", help="Optional path to write the markdown report.")
    parser.add_argument("--json-output", default="", help="Optional path to write the JSON report.")
    args = parser.parse_args(argv)

    predictions = pd.read_csv(args.predictions)
    report = evaluate_fusion(
        predictions,
        labels=_split(args.labels),
        thresholds=args.threshold,
        group_column=args.group_column,
        identity_labels=_split(args.identity_labels) or None,
    )

    rendered = report.render()
    print(rendered)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
