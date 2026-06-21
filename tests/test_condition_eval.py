import json

import pandas as pd

from fbm_multimodal.cli import main
from fbm_multimodal.condition_eval import evaluate_conditions


def _condition_rows(condition, group, truths, probs):
    rows = []
    for idx, (truth, prob) in enumerate(zip(truths, probs)):
        rows.append(
            {
                "condition": condition,
                "chip_id": f"{condition}-{group}-{idx}",
                "eval_group": group,
                "true_a": truth[0],
                "true_b": truth[1],
                "prob_a": prob[0],
                "prob_b": prob[1],
            }
        )
    return rows


def test_evaluate_conditions_flags_kpi_gate_and_required_accuracy():
    rows = []
    rows += _condition_rows(
        "passes-kpi",
        "real_single",
        truths=[(1, 0), (1, 0), (0, 1), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1)],
        probs=[(0.9, 0.1), (0.8, 0.2), (0.2, 0.7), (0.3, 0.8), (0.7, 0.2), (0.4, 0.7), (0.8, 0.1), (0.7, 0.2)],
    )
    rows += _condition_rows(
        "passes-kpi",
        "real_composite",
        truths=[(1, 1), (1, 1), (1, 1), (1, 1)],
        probs=[(0.8, 0.7), (0.7, 0.6), (0.9, 0.8), (0.8, 0.4)],
    )
    rows += _condition_rows(
        "passes-minima-only",
        "real_single",
        truths=[(1, 0), (1, 0), (0, 1), (0, 1), (1, 0)],
        probs=[(0.9, 0.1), (0.8, 0.2), (0.2, 0.7), (0.3, 0.8), (0.4, 0.7)],
    )
    rows += _condition_rows(
        "passes-minima-only",
        "real_composite",
        truths=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
        probs=[(0.8, 0.7), (0.7, 0.6), (0.9, 0.8), (0.8, 0.4), (0.4, 0.9)],
    )

    summary = evaluate_conditions(pd.DataFrame(rows), labels=["a", "b"])

    passing = summary.set_index("condition").loc["passes-kpi"]
    minima_only = summary.set_index("condition").loc["passes-minima-only"]
    assert passing["single_subset_accuracy"] == 0.875
    assert passing["composite_subset_accuracy"] == 0.75
    assert passing["kpi_product"] == 0.65625
    assert bool(passing["meets_all_targets"]) is True
    assert minima_only["single_subset_accuracy"] == 0.8
    assert minima_only["composite_subset_accuracy"] == 0.6
    assert minima_only["kpi_product"] == 0.48
    assert bool(minima_only["meets_single_target"]) is True
    assert bool(minima_only["meets_composite_target"]) is True
    assert bool(minima_only["meets_kpi_target"]) is False
    assert minima_only["required_composite_for_kpi_at_single"] == 0.8125


def test_evaluate_conditions_reports_synthetic_to_real_composite_gap():
    rows = []
    rows += _condition_rows(
        "image-synth",
        "real_single",
        truths=[(1, 0), (0, 1)],
        probs=[(0.9, 0.1), (0.1, 0.9)],
    )
    rows += _condition_rows(
        "image-synth",
        "real_composite",
        truths=[(1, 1), (1, 1)],
        probs=[(0.8, 0.7), (0.8, 0.4)],
    )
    rows += _condition_rows(
        "image-synth",
        "synthetic_composite",
        truths=[(1, 1), (1, 1)],
        probs=[(0.8, 0.7), (0.9, 0.8)],
    )

    summary = evaluate_conditions(pd.DataFrame(rows), labels=["a", "b"])

    result = summary.iloc[0]
    assert result["real_synthetic_composite_gap"] == 0.5
    assert result["synthetic_composite_subset_accuracy"] == 1.0


def test_cli_evaluate_conditions_writes_summary_csv_and_json(tmp_path, capsys):
    predictions_path = tmp_path / "predictions.csv"
    output_path = tmp_path / "summary.csv"
    rows = []
    rows += _condition_rows(
        "fusion",
        "real_single",
        truths=[(1, 0), (0, 1)],
        probs=[(0.9, 0.1), (0.1, 0.9)],
    )
    rows += _condition_rows(
        "fusion",
        "real_composite",
        truths=[(1, 1), (1, 1)],
        probs=[(0.8, 0.7), (0.9, 0.8)],
    )
    pd.DataFrame(rows).to_csv(predictions_path, index=False)

    exit_code = main(
        [
            "evaluate-conditions",
            "--predictions",
            str(predictions_path),
            "--labels",
            "a,b",
            "--output",
            str(output_path),
        ]
    )

    stdout = json.loads(capsys.readouterr().out)
    summary = pd.read_csv(output_path)
    assert exit_code == 0
    assert stdout["best_condition_by_kpi"] == "fusion"
    assert summary.loc[0, "meets_all_targets"]
