import numpy as np
import pandas as pd

from fbm_multimodal.fusion.fusion_eval import (
    evaluate_fusion,
    modality_contribution,
    run_leakage_checks,
    wilson_ci,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_wilson_ci_handles_empty_and_full():
    assert wilson_ci(0, 0) == (0.0, 1.0)
    low, high = wilson_ci(5, 5)
    assert high == 1.0
    assert 0.0 < low < 1.0


def test_synthetic_rows_excluded_from_tabular_and_fusion_heads():
    # synthetic_composite rows have NaN tabular/fusion probs -> those heads skip them.
    rows = [
        # real single: all heads available
        {"eval_group": "real_single", "true_a": 1, "true_b": 0,
         "image_prob_a": 0.9, "image_prob_b": 0.1,
         "tabular_prob_a": 0.8, "tabular_prob_b": 0.2,
         "fusion_prob_a": 0.95, "fusion_prob_b": 0.05},
        # synthetic composite: only image available
        {"eval_group": "synthetic_composite", "true_a": 1, "true_b": 1,
         "image_prob_a": 0.7, "image_prob_b": 0.6,
         "tabular_prob_a": np.nan, "tabular_prob_b": np.nan,
         "fusion_prob_a": np.nan, "fusion_prob_b": np.nan},
    ]
    report = evaluate_fusion(_frame(rows), labels=["a", "b"])

    # image head is available on both groups
    assert report.head_group_accuracy["image_only"]["real_single"].support == 1
    assert report.head_group_accuracy["image_only"]["synthetic_composite"].support == 1
    # tabular/fusion heads have zero support on synthetic group
    assert report.head_group_accuracy["tabular_only"]["synthetic_composite"].support == 0
    assert report.head_group_accuracy["fusion"]["synthetic_composite"].support == 0


def test_kpi_product_is_single_times_composite_per_head():
    rows = [
        {"eval_group": "real_single", "true_a": 1, "true_b": 0,
         "fusion_prob_a": 0.9, "fusion_prob_b": 0.1},   # correct
        {"eval_group": "real_single", "true_a": 0, "true_b": 1,
         "fusion_prob_a": 0.1, "fusion_prob_b": 0.9},   # correct -> single acc = 1.0
        {"eval_group": "real_composite", "true_a": 1, "true_b": 1,
         "fusion_prob_a": 0.9, "fusion_prob_b": 0.9},   # correct
        {"eval_group": "real_composite", "true_a": 1, "true_b": 1,
         "fusion_prob_a": 0.9, "fusion_prob_b": 0.1},   # wrong -> composite acc = 0.5
    ]
    report = evaluate_fusion(_frame(rows), labels=["a", "b"])
    kpi = report.kpi_by_head["fusion"]
    assert kpi["single_acc"] == 1.0
    assert kpi["composite_acc"] == 0.5
    assert kpi["kpi_product"] == 0.5


def test_real_all_group_combines_real_single_and_real_composite_only():
    rows = [
        {"eval_group": "real_single", "true_a": 1, "true_b": 0,
         "fusion_prob_a": 0.9, "fusion_prob_b": 0.1},
        {"eval_group": "real_composite", "true_a": 1, "true_b": 1,
         "fusion_prob_a": 0.9, "fusion_prob_b": 0.9},
        {"eval_group": "synthetic_composite", "true_a": 1, "true_b": 1,
         "fusion_prob_a": 0.1, "fusion_prob_b": 0.1},
    ]
    report = evaluate_fusion(_frame(rows), labels=["a", "b"])

    assert report.head_group_accuracy["fusion"]["real_all"].support == 2
    assert report.head_group_accuracy["fusion"]["real_all"].accuracy == 1.0
    assert report.head_group_accuracy["fusion"]["synthetic_composite"].support == 1


def test_fusion_prob_alias_accepts_bare_prob_columns():
    # A CSV produced for the core evaluate-conditions CLI (prob_<label>) still works.
    rows = [
        {"eval_group": "real_composite", "true_a": 1, "true_b": 1,
         "prob_a": 0.9, "prob_b": 0.9},
    ]
    report = evaluate_fusion(_frame(rows), labels=["a", "b"])
    assert "fusion" in report.heads_present
    assert report.kpi_by_head["fusion"]["composite_acc"] == 1.0


def test_collapse_diagnostic_flags_when_fusion_ignores_tabular():
    # Build composite rows where tabular rescues image but fusion follows image.
    rows = []
    for _ in range(10):
        rows.append(
            {"eval_group": "real_composite", "true_a": 1, "true_b": 1,
             # image is wrong on b
             "image_prob_a": 0.9, "image_prob_b": 0.1,
             # tabular is right on both
             "tabular_prob_a": 0.9, "tabular_prob_b": 0.9,
             # fusion copies the (wrong) image answer
             "fusion_prob_a": 0.9, "fusion_prob_b": 0.1}
        )
    report = evaluate_fusion(_frame(rows), labels=["a", "b"])
    diag = report.collapse_diagnostic
    assert diag["available"] == 1.0
    assert diag["tabular_rescue_candidates"] == 10.0
    assert diag["fusion_followed_tabular"] == 0.0
    assert diag["fusion_follow_rate"] == 0.0
    assert any("collapse" in w for w in report.warnings)


def test_identity_slice_reports_tabular_advantage():
    # On identity label b, tabular separates but image does not.
    rows = [
        {"eval_group": "real_composite", "true_a": 0, "true_b": 1,
         "image_prob_a": 0.4, "image_prob_b": 0.4,      # image wrong on b
         "tabular_prob_a": 0.1, "tabular_prob_b": 0.9,  # tabular right
         "fusion_prob_a": 0.1, "fusion_prob_b": 0.9},
        {"eval_group": "real_composite", "true_a": 0, "true_b": 1,
         "image_prob_a": 0.4, "image_prob_b": 0.3,
         "tabular_prob_a": 0.2, "tabular_prob_b": 0.8,
         "fusion_prob_a": 0.2, "fusion_prob_b": 0.8},
    ]
    report = evaluate_fusion(_frame(rows), labels=["a", "b"], identity_labels=["b"])
    assert report.identity_slice["n"] == 2.0
    assert report.identity_slice["tabular_minus_image"] > 0


def test_modality_contribution_measures_tabular_drop():
    # A model that only works when tabular is present.
    def predict_fn(images, tabular):
        # prob = tabular value directly; with null tabular -> all zeros -> wrong
        return np.asarray(tabular, dtype=float)

    images = np.zeros((4, 3))
    tabular = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    out = modality_contribution(predict_fn, images, tabular, y_true, thresholds=0.5)
    assert out["subset_acc_with_tabular"] == 1.0
    assert out["subset_acc_tabular_ablated"] == 0.0
    assert out["tabular_contribution"] == 1.0


def test_leakage_checks_flag_common_wl_catboost_and_pseudo_label_risks():
    predictions = pd.DataFrame(
        {
            "sample_id": ["syn-as-real"],
            "eval_group": ["real_composite"],
            "is_synthetic": [True],
        }
    )

    warnings = run_leakage_checks(
        predictions,
        tensorizer_fit_sample_ids={"train-a", "valid-leak"},
        train_real_sample_ids={"train-a"},
        catboost_metadata={"train_prediction_mode": "in_fold", "synthetic_excluded": False},
        pseudo_labeling_enabled=True,
    )

    assert any("WL baseline" in warning for warning in warnings)
    assert any("CatBoost train logits" in warning for warning in warnings)
    assert any("Synthetic samples are not excluded" in warning for warning in warnings)
    assert any("Pseudo-labeling is enabled" in warning for warning in warnings)
    assert any("official metric" in warning for warning in warnings)
