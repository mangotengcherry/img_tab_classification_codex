import json

import numpy as np
import pandas as pd

from fbm_multimodal.fusion.real_data import (
    build_fusion_manifest,
    load_eds_tabular,
    load_fbm_tensor_dataset,
)


def test_load_fbm_tensor_dataset_validates_manifest_and_label_map(tmp_path):
    fbm_dir = tmp_path / "fbm_tensor"
    fbm_dir.mkdir()
    np.save(fbm_dir / "fbm_images.npy", np.zeros((2, 4, 4), dtype=np.float32))
    (fbm_dir / "label_map.json").write_text(
        json.dumps({"label_columns": ["label_a", "label_b"], "label_names": ["a", "b"]}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "row_idx": [0, 1],
            "sample_id": ["s1", "s2"],
            "split": ["train", "test"],
            "eval_group": ["real_single", "real_composite"],
            "label_a": [1, 0],
            "label_b": [0, 1],
        }
    ).to_csv(fbm_dir / "fbm_manifest.csv", index=False)

    images, manifest, label_map = load_fbm_tensor_dataset(fbm_dir)

    assert images.shape == (2, 4, 4)
    assert manifest["sample_id"].tolist() == ["s1", "s2"]
    assert label_map["label_columns"] == ["label_a", "label_b"]


def test_build_fusion_manifest_marks_missing_eds_and_rejects_label_mismatch():
    fbm_manifest = pd.DataFrame(
        {
            "row_idx": [0, 1],
            "sample_id": ["s1", "s2"],
            "split": ["train", "test"],
            "eval_group": ["real_single", "real_composite"],
            "label_a": [1, 0],
        }
    )
    eds = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "split": ["train"],
            "eval_group": ["real_single"],
            "label_a": [1],
            "EDS_RD_WL000": [12.0],
        }
    )

    manifest = build_fusion_manifest(fbm_manifest, eds, label_columns=["label_a"])

    assert manifest["has_fbm_image"].tolist() == [1, 1]
    assert manifest["has_eds_tabular"].tolist() == [1, 0]
    assert manifest["has_wl_map"].tolist() == [0, 0]
    assert manifest["has_catboost_logits"].tolist() == [0, 0]

    mismatched = eds.assign(label_a=[0])
    try:
        build_fusion_manifest(fbm_manifest, mismatched, label_columns=["label_a"])
    except ValueError as exc:
        assert "label mismatch" in str(exc)
    else:
        raise AssertionError("expected label mismatch error")


def test_load_eds_tabular_reads_csv_and_rejects_duplicate_sample_ids(tmp_path):
    path = tmp_path / "eds_tabular.csv"
    pd.DataFrame(
        {
            "sample_id": ["dup", "dup"],
            "split": ["train", "train"],
            "eval_group": ["real_single", "real_single"],
            "label_a": [1, 1],
            "EDS_RD_WL000": [1.0, 2.0],
        }
    ).to_csv(path, index=False)

    try:
        load_eds_tabular(path)
    except ValueError as exc:
        assert "duplicate sample_id" in str(exc)
    else:
        raise AssertionError("expected duplicate sample_id error")
