import numpy as np

from fbm_multimodal.training.train_catboost_oof import _infer_feature_columns, train_catboost_oof_logits


class _MemorizingBinaryEstimator:
    fitted_ids: list[set[int]] = []

    def __init__(self) -> None:
        self.train_ids: set[int] = set()

    def fit(self, x, y):
        self.train_ids = set(x[:, 0].astype(int).tolist())
        self.fitted_ids.append(set(self.train_ids))
        return self

    def predict_proba(self, x):
        ids = x[:, 0].astype(int)
        positive = np.array([0.99 if sample_id in self.train_ids else 0.01 for sample_id in ids])
        return np.column_stack([1.0 - positive, positive])


def test_catboost_oof_train_logits_are_out_of_fold_and_exclude_synthetic(tmp_path):
    _MemorizingBinaryEstimator.fitted_ids = []
    x = np.array([[0.0], [1.0], [2.0], [3.0], [100.0], [200.0]])
    y = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0],
        ]
    )

    result = train_catboost_oof_logits(
        x,
        y,
        sample_ids=["tr0", "tr1", "tr2", "tr3", "syn", "val"],
        split=["train", "train", "train", "train", "train", "valid"],
        is_synthetic=[False, False, False, False, True, False],
        num_folds=2,
        output_dir=tmp_path,
        model_factory=lambda class_index, random_seed: _MemorizingBinaryEstimator(),
    )

    train_logits = result.train_oof_logits
    assert train_logits["sample_id"].tolist() == ["tr0", "tr1", "tr2", "tr3"]
    assert (train_logits[["cat_logit_0", "cat_logit_1"]] < 0).all().all()
    assert all(100 not in ids for ids in _MemorizingBinaryEstimator.fitted_ids)
    assert "valid" in result.split_logits
    assert result.split_logits["valid"]["sample_id"].tolist() == ["val"]
    assert result.metadata["train_prediction_mode"] == "oof"
    assert result.metadata["synthetic_excluded"] is True
    assert (tmp_path / "metadata.json").exists()
    assert any((tmp_path / "models").glob("class_*_fold_*.pkl"))


def test_catboost_cli_feature_inference_excludes_labels_and_metadata():
    assert _infer_feature_columns(
        ["sample_id", "split", "is_synthetic", "label_a", "label_b", "EDS_RD_WL000"],
        sample_id_column="sample_id",
        split_column="split",
        synthetic_column="is_synthetic",
        label_columns=["label_a", "label_b"],
    ) == ["EDS_RD_WL000"]
