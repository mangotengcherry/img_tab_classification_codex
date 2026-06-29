import numpy as np

from fbm_multimodal.fusion.data import DEFAULT_WL_CHANNELS, FusionDataset, generate_dataset
from fbm_multimodal.fusion.model import ClasswiseGatedResidualFusion, FusionMLP, WLResidualCatBoostFusionMLP


def _toy(seed: int = 0):
    """Two labels: label 0 separable by image, label 1 separable by tabular."""
    rng = np.random.default_rng(seed)
    n = 240
    y = rng.integers(0, 2, size=(n, 2))
    images = np.zeros((n, 8))
    images[:, 0] = y[:, 0] * 5.0 + rng.normal(0, 0.2, n)   # image encodes label 0
    tabular = np.zeros((n, 6))
    tabular[:, 0] = y[:, 1] * 5.0 + rng.normal(0, 0.2, n)   # tabular encodes label 1
    has_tab = np.ones(n, dtype=bool)
    # make a third of rows synthetic (image only)
    has_tab[: n // 3] = False
    tabular[~has_tab] = np.nan
    return images, tabular, y, has_tab


def test_training_reduces_total_loss():
    images, tabular, y, has_tab = _toy()
    model = FusionMLP(hidden=16, lr=5e-3, epochs=120, dropout_p=0.2, seed=1)
    model.fit(images, tabular, y, has_tab)
    assert model.history["loss_total"][-1] < model.history["loss_total"][0]


def test_fusion_uses_both_modalities_on_toy():
    images, tabular, y, has_tab = _toy()
    model = FusionMLP(hidden=16, lr=5e-3, epochs=200, dropout_p=0.2, seed=1)
    model.fit(images, tabular, y, has_tab)
    heads = model.predict_heads(images, tabular, has_tab)
    real = has_tab
    fusion_pred = (heads["fusion"][real] >= 0.5).astype(int)
    acc = (fusion_pred == y[real]).all(axis=1).mean()
    # fusion sees both label-0 (image) and label-1 (tabular) signals
    assert acc > 0.85


def test_predictions_are_nan_for_rows_without_tabular():
    images, tabular, y, has_tab = _toy()
    model = FusionMLP(hidden=8, lr=5e-3, epochs=20, seed=1)
    model.fit(images, tabular, y, has_tab)
    heads = model.predict_heads(images, tabular, has_tab)
    assert np.isnan(heads["tabular"][~has_tab]).all()
    assert np.isnan(heads["fusion"][~has_tab]).all()
    assert not np.isnan(heads["image"]).any()  # image head always defined


def test_generated_dataset_shapes_and_asymmetry():
    ds = generate_dataset(seed=3)
    assert ds.images.shape[1:] == (128, 46)
    assert ds.tabular.shape[1] == 201
    assert ds.labels.shape[1] == len(ds.label_names)
    # synthetic composite rows must have NO tabular
    synth = ds.eval_group == "synthetic_composite"
    assert np.isnan(ds.tabular[synth]).all()
    # real rows must HAVE tabular
    real = np.isin(ds.eval_group, np.array(["real_single", "real_composite"]))
    assert not np.isnan(ds.tabular[real]).any()


def test_dataset_uses_explicit_wl_and_catboost_masks_when_values_are_zero_filled():
    images = np.zeros((2, 2, 2), dtype=float)
    tabular = np.full((2, 3), np.nan)
    labels = np.zeros((2, 2), dtype=int)
    wl_maps = np.zeros((2, len(DEFAULT_WL_CHANNELS), 2, 1), dtype=float)
    observed_idx = DEFAULT_WL_CHANNELS.index("observed_mask")
    wl_maps[0, observed_idx, 0, 0] = 1.0
    catboost_logits = np.zeros((2, 2), dtype=float)

    ds = FusionDataset(
        images=images,
        tabular=tabular,
        labels=labels,
        eval_group=np.array(["real_single", "synthetic_composite"]),
        chip_id=["real", "synthetic"],
        split=np.array(["train", "train"]),
        label_names=["a", "b"],
        identity_labels=[],
        wl_maps=wl_maps,
        catboost_logits=catboost_logits,
        has_catboost_logits_array=np.array([1.0, 0.0]),
    )

    np.testing.assert_array_equal(ds.has_wl_map, np.array([True, False]))
    np.testing.assert_array_equal(ds.has_catboost_logits, np.array([True, False]))
    np.testing.assert_array_equal(ds.is_synthetic, np.array([False, True]))
    assert ds.sample_type.tolist() == ["real_single", "synthetic_composite"]


def test_classwise_gated_residual_fusion_respects_modality_masks():
    fusion = ClasswiseGatedResidualFusion(
        num_classes=2,
        wl_gates=np.array([0.5, 2.0]),
        catboost_gates=np.array([1.0, 0.25]),
    )
    fbm_logits = np.array([[1.0, 1.0], [1.0, 1.0]])
    wl_logits = np.array([[2.0, 3.0], [10.0, 10.0]])
    catboost_logits = np.array([[4.0, 8.0], [4.0, 8.0]])

    combined = fusion.combine_logits(
        fbm_logits,
        wl_logits=wl_logits,
        has_wl_map=np.array([1.0, 0.0]),
        catboost_logits=catboost_logits,
        has_catboost_logits=np.array([1.0, 1.0]),
    )

    np.testing.assert_allclose(combined[0], [6.0, 9.0])
    np.testing.assert_allclose(combined[1], [5.0, 3.0])


def test_wl_residual_catboost_model_trains_and_masks_unavailable_fusion_rows():
    rng = np.random.default_rng(11)
    n = 120
    y = rng.integers(0, 2, size=(n, 2))
    images = np.zeros((n, 4), dtype=float)
    images[:, 0] = y[:, 0] * 4.0 + rng.normal(0, 0.2, n)
    wl_maps = np.zeros((n, 1, 2, 1), dtype=float)
    wl_maps[:, 0, 0, 0] = y[:, 1] * 4.0 + rng.normal(0, 0.2, n)
    has_wl = np.ones(n, dtype=float)
    has_wl[:10] = 0.0
    wl_loss_weight = has_wl.copy()
    catboost_logits = np.zeros((n, 2), dtype=float)
    catboost_logits[:, 1] = np.where(y[:, 1] == 1, 2.0, -2.0)
    has_cat = np.ones(n, dtype=float)
    has_cat[10:20] = 0.0
    has_wl[:5] = 0.0
    has_cat[:5] = 0.0

    model = WLResidualCatBoostFusionMLP(hidden=12, lr=5e-3, epochs=140, seed=3)
    model.fit(
        images,
        wl_maps,
        y,
        has_wl_map=has_wl,
        wl_loss_weight=wl_loss_weight,
        catboost_logits=catboost_logits,
        has_catboost_logits=has_cat,
    )

    assert model.history["loss_total"][-1] < model.history["loss_total"][0]
    heads = model.predict_heads(
        images,
        wl_maps,
        has_wl_map=has_wl,
        catboost_logits=catboost_logits,
        has_catboost_logits=has_cat,
    )
    assert np.isnan(heads["fusion"][:5]).all()
    assert not np.isnan(heads["fusion"][20:]).any()
    fusion_pred = (heads["fusion"][20:] >= 0.5).astype(int)
    assert (fusion_pred == y[20:]).all(axis=1).mean() > 0.8
