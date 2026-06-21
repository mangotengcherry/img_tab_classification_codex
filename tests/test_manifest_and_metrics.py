import pandas as pd

from fbm_multimodal.manifest import (
    assert_no_synthetic_or_pseudo_in_real_eval,
    label_columns,
)
from fbm_multimodal.metrics import (
    class_pair_metrics,
    compute_multilabel_metrics,
    optimize_class_thresholds,
    synthetic_to_real_gap,
)


def test_label_columns_exclude_metadata_and_msr_features():
    frame = pd.DataFrame(
        columns=[
            "chip_id",
            "image_path",
            "is_real",
            "is_synthetic",
            "split",
            "MSR_000",
            "defect_a",
            "defect_b",
        ]
    )

    assert label_columns(frame) == ["defect_a", "defect_b"]


def test_real_eval_guard_rejects_synthetic_and_pseudo_rows():
    frame = pd.DataFrame(
        {
            "chip_id": ["real-1", "synthetic-1", "pseudo-1"],
            "is_synthetic": [False, True, False],
            "is_pseudo_labeled": [False, False, True],
        }
    )

    problems = assert_no_synthetic_or_pseudo_in_real_eval(frame)

    assert problems == ["synthetic-1", "pseudo-1"]


def test_compute_multilabel_metrics_includes_kpi_and_per_class_f1():
    y_true = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]})
    y_prob = pd.DataFrame({"a": [0.9, 0.2, 0.8], "b": [0.1, 0.7, 0.4]})

    metrics = compute_multilabel_metrics(y_true, y_prob, thresholds={"a": 0.5, "b": 0.5})

    assert metrics.subset_accuracy == 2 / 3
    assert metrics.hamming_accuracy == 5 / 6
    assert metrics.per_class["a"]["f1"] == 1.0
    assert metrics.per_class["b"]["recall"] == 0.5


def test_optimize_class_thresholds_uses_validation_probabilities():
    y_true = pd.DataFrame({"a": [1, 1, 0, 0], "b": [1, 0, 1, 0]})
    y_prob = pd.DataFrame({"a": [0.8, 0.6, 0.55, 0.1], "b": [0.9, 0.45, 0.4, 0.2]})

    thresholds = optimize_class_thresholds(y_true, y_prob, candidates=[0.5, 0.6, 0.7])

    assert thresholds["a"] == 0.6
    assert thresholds["b"] == 0.5


def test_class_pair_metrics_and_synthetic_gap_are_reported_separately():
    y_true = pd.DataFrame({"a": [1, 1, 0], "b": [1, 1, 1], "c": [0, 0, 0]})
    y_pred_real = pd.DataFrame({"a": [1, 0, 0], "b": [1, 1, 1], "c": [0, 0, 0]})
    y_pred_synth = pd.DataFrame({"a": [1, 1, 0], "b": [1, 1, 1], "c": [0, 0, 0]})

    real_pairs = class_pair_metrics(y_true, y_pred_real)
    synth_pairs = class_pair_metrics(y_true, y_pred_synth)
    gap = synthetic_to_real_gap(real_pairs, synth_pairs)

    assert real_pairs["a+b"]["subset_accuracy"] == 0.5
    assert synth_pairs["a+b"]["subset_accuracy"] == 1.0
    assert gap["a+b"]["gap"] == 0.5
