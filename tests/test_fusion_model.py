from itertools import combinations

import numpy as np

from fbm_multimodal.fusion.data import generate_dataset
from fbm_multimodal.fusion.model import FusionMLP
from fbm_multimodal.fusion.visualize import plot_domain_pattern_stress_gallery, plot_pattern_gallery


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


def test_generated_composites_cover_all_label_pairs_even_when_small():
    ds = generate_dataset(
        seed=7,
        n_real_single_train=12,
        n_real_composite_train=6,
        n_synth_composite_train=6,
        n_real_single_test=12,
        n_real_composite_test=6,
        n_synth_composite_test=6,
    )
    expected_pairs = set(combinations(range(len(ds.label_names)), 2))

    for split in ["train", "test"]:
        for group in ["real_composite", "synthetic_composite"]:
            mask = (ds.split == split) & (ds.eval_group == group)
            seen = {
                tuple(np.flatnonzero(row))
                for row in ds.labels[mask]
            }
            assert seen == expected_pairs


def test_pattern_gallery_visualization_is_written(tmp_path):
    ds = generate_dataset(
        seed=5,
        n_real_single_train=16,
        n_real_composite_train=6,
        n_synth_composite_train=6,
        n_real_single_test=8,
        n_real_composite_test=6,
        n_synth_composite_test=6,
    )
    out = plot_pattern_gallery(ds, tmp_path / "pattern_gallery.png")

    assert out.exists()
    assert out.stat().st_size > 0


def test_domain_pattern_stress_gallery_visualization_is_written(tmp_path):
    out = plot_domain_pattern_stress_gallery(tmp_path / "domain_pattern_stress_gallery.png")

    assert out.exists()
    assert out.stat().st_size > 0
