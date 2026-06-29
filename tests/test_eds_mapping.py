import pandas as pd

from fbm_multimodal.eds_mapping import (
    catboost_feature_columns,
    validate_eds_wordline_map,
    wide_eds_to_wl_measurements,
)


def test_validate_eds_wordline_map_accepts_minimal_schema_and_derives_flags():
    mapping = pd.DataFrame(
        {
            "feature_name": ["EDS_RD_WL000", "EDS_GLOBAL_IDDQ"],
            "eds_step": ["READ", "IDDQ"],
            "eds_item": ["RD_LEAK", "IDDQ_TOTAL"],
            "wordline_position": [0, None],
        }
    )

    validated = validate_eds_wordline_map(
        mapping,
        eds_columns=["sample_id", "EDS_RD_WL000", "EDS_GLOBAL_IDDQ"],
    )

    assert validated["test_method"].tolist() == ["READ", "IDDQ"]
    assert validated["test_item"].tolist() == ["RD_LEAK", "IDDQ_TOTAL"]
    assert validated["include_in_wl_map"].tolist() == [1, 0]
    assert validated["include_in_catboost"].tolist() == [1, 1]


def test_wide_eds_to_wl_measurements_excludes_global_items_and_flips_low_bad_values():
    eds = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "split": ["train"],
            "eval_group": ["real_single"],
            "is_synthetic": [False],
            "EDS_RD_WL000": [10.0],
            "EDS_MARGIN_WL001": [3.0],
            "EDS_GLOBAL_IDDQ": [99.0],
        }
    )
    mapping = pd.DataFrame(
        {
            "feature_name": ["EDS_RD_WL000", "EDS_MARGIN_WL001", "EDS_GLOBAL_IDDQ"],
            "eds_step": ["READ", "READ", "IDDQ"],
            "eds_item": ["RD_LEAK", "RD_MARGIN", "IDDQ_TOTAL"],
            "wordline_position": [0, "WL001", None],
            "value_direction": ["high_bad", "low_bad", "high_bad"],
        }
    )

    measurements = wide_eds_to_wl_measurements(eds, mapping)

    assert measurements["feature_name"].tolist() == ["EDS_RD_WL000", "EDS_MARGIN_WL001"]
    assert measurements["test_method"].tolist() == ["READ", "READ"]
    assert measurements["wordline"].tolist() == [0, 1]
    assert measurements["value"].tolist() == [10.0, -3.0]
    assert measurements["is_synthetic"].tolist() == [False, False]


def test_catboost_feature_columns_uses_mapping_and_excludes_labels_metadata_and_synthetic_only_flags():
    eds = pd.DataFrame(
        columns=[
            "sample_id",
            "split",
            "eval_group",
            "label_a",
            "EDS_RD_WL000",
            "EDS_GLOBAL_IDDQ",
        ]
    )
    mapping = pd.DataFrame(
        {
            "feature_name": ["EDS_RD_WL000", "EDS_GLOBAL_IDDQ"],
            "eds_step": ["READ", "IDDQ"],
            "eds_item": ["RD_LEAK", "IDDQ_TOTAL"],
            "wordline_position": [0, None],
            "include_in_catboost": [1, 0],
        }
    )

    assert catboost_feature_columns(eds, mapping, label_columns=["label_a"]) == ["EDS_RD_WL000"]
