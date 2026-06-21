import pandas as pd

from fbm_multimodal.measurement import MeasurementMap


def test_mapping_coverage_reports_missing_manifest_features():
    mapping = MeasurementMap.from_frame(
        pd.DataFrame(
            {
                "feature_name": ["MSR_000", "MSR_002"],
                "measurement_condition": ["A", "A"],
                "measurement_type": ["leakage", "leakage"],
                "wl_index": [100, 0],
                "physical_region": ["top", "bottom"],
                "physical_order": [2, 0],
            }
        )
    )

    coverage = mapping.coverage(["MSR_000", "MSR_001", "MSR_002"])

    assert coverage.total_features == 3
    assert coverage.mapped_features == 2
    assert coverage.missing_features == ["MSR_001"]
    assert coverage.coverage_ratio == 2 / 3


def test_physical_order_uses_mapping_not_msr_suffix():
    mapping = MeasurementMap.from_frame(
        pd.DataFrame(
            {
                "feature_name": ["MSR_000", "MSR_001", "MSR_002"],
                "measurement_condition": ["A", "A", "A"],
                "measurement_type": ["leakage", "leakage", "leakage"],
                "wl_index": [10, 0, 5],
                "physical_region": ["top", "bottom", "middle"],
                "physical_order": [2, 0, 1],
            }
        )
    )

    assert mapping.features_by_physical_order() == ["MSR_001", "MSR_002", "MSR_000"]


def test_metadata_features_are_region_condition_aggregates():
    mapping = MeasurementMap.from_frame(
        pd.DataFrame(
            {
                "feature_name": ["MSR_A", "MSR_B", "MSR_C"],
                "measurement_condition": ["read", "read", "stress"],
                "measurement_type": ["leakage", "leakage", "leakage"],
                "wl_index": [None, None, None],
                "physical_region": ["top", "top", "bottom"],
                "physical_order": [None, None, None],
            }
        )
    )
    row = pd.Series({"MSR_A": 0.0, "MSR_B": 10.0, "MSR_C": 4.0})

    features = mapping.aggregate_row(row)

    assert features["region=top__mean"] == 5.0
    assert features["region=top__max"] == 10.0
    assert features["region=bottom__mean"] == 4.0
    assert features["condition=read__mean"] == 5.0
    assert features["condition=stress__max"] == 4.0
