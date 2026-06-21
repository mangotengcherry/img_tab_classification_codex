import pandas as pd

from fbm_multimodal.baselines import SklearnMultiLabelBaseline
from fbm_multimodal.features import build_tabular_feature_frame, build_late_fusion_frame
from fbm_multimodal.measurement import MeasurementMap


def test_tabular_feature_frame_uses_metadata_aggregates_and_stable_raw_names():
    manifest = pd.DataFrame(
        {
            "MSR_000": [10.0],
            "MSR_001": [2.0],
            "MSR_002": [5.0],
        }
    )
    mapping = MeasurementMap.from_frame(
        pd.DataFrame(
            {
                "feature_name": ["MSR_000", "MSR_001", "MSR_002"],
                "measurement_condition": ["read", "read", "stress"],
                "measurement_type": ["leakage", "leakage", "leakage"],
                "physical_region": ["top", "bottom", "middle"],
                "physical_order": [2, 0, 1],
            }
        )
    )

    features = build_tabular_feature_frame(manifest, mapping)

    assert list(features.columns[:3]) == ["raw::MSR_000", "raw::MSR_001", "raw::MSR_002"]
    assert features.loc[0, "region=top__mean"] == 10.0
    assert features.loc[0, "condition=read__mean"] == 6.0


def test_late_fusion_frame_keeps_image_and_tabular_probabilities_separate():
    image_prob = pd.DataFrame({"a": [0.9, 0.1], "b": [0.2, 0.8]})
    tabular_prob = pd.DataFrame({"a": [0.7, 0.4], "b": [0.3, 0.6]})

    features = build_late_fusion_frame(image_prob, tabular_prob)

    assert list(features.columns) == [
        "image_prob_a",
        "image_prob_b",
        "tabular_prob_a",
        "tabular_prob_b",
        "abs_diff_a",
        "abs_diff_b",
    ]
    assert features.loc[0, "abs_diff_a"] == 0.2


def test_sklearn_multilabel_baseline_returns_label_aligned_probabilities():
    x_train = pd.DataFrame(
        {
            "feature_1": [0.0, 0.1, 1.0, 1.1],
            "feature_2": [1.0, 0.9, 0.0, 0.1],
        }
    )
    y_train = pd.DataFrame(
        {
            "defect_a": [0, 0, 1, 1],
            "defect_b": [1, 1, 0, 0],
        }
    )
    model = SklearnMultiLabelBaseline(random_state=7)

    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_train)

    assert list(probabilities.columns) == ["defect_a", "defect_b"]
    assert probabilities.shape == (4, 2)
    assert probabilities["defect_a"].iloc[-1] > probabilities["defect_a"].iloc[0]
