import numpy as np

from fbm_multimodal.fusion.fbm_patterns import (
    IMAGE_SHAPE,
    binarize_fbm,
    effective_rank,
    eigenimage_norm_features,
    image_feature_matrix,
    nonneg_factorize,
    paint_cluster,
    paint_single_bit_scatter,
    paint_vertical_line,
)


def test_binarize_uses_grade_three_threshold():
    grades = np.array([[0.0, 2.9, 3.0, 8.0]])
    binary = binarize_fbm(grades)
    np.testing.assert_array_equal(binary, [[0.0, 0.0, 1.0, 1.0]])


def test_eigenimage_norms_are_descending_and_padded():
    rng = np.random.default_rng(0)
    binary = (rng.random((128, 46)) > 0.5).astype(float)
    norms = eigenimage_norm_features(binary, k=16)
    assert norms.shape == (16,)
    assert np.all(np.diff(norms) <= 1e-9)  # non-increasing


def test_structured_pattern_is_low_rank_vs_random_scatter():
    # a vertical line is rank-1 (structured); random scatter is high-rank.
    line = np.zeros((128, 46))
    paint_vertical_line(line, col=23, width=2, grade=6.0)
    line_bin = binarize_fbm(line)

    rng = np.random.default_rng(1)
    scatter = (rng.random((128, 46)) < 0.05).astype(float)  # random single-bit

    assert effective_rank(line_bin, energy=0.9) <= 2
    assert effective_rank(scatter, energy=0.9) > effective_rank(line_bin, energy=0.9)
    # first norm dominates for the structured line, not for the scatter
    line_norms = eigenimage_norm_features(line_bin)
    assert line_norms[0] > 5 * (line_norms[1] + 1e-9)


def test_nonneg_factorize_reconstructs_better_than_zero():
    line = np.zeros((128, 46))
    paint_cluster(line, center=(64, 23), half=(10, 6), grade=6.0)
    binary = binarize_fbm(line)
    w, h, norms = nonneg_factorize(binary, k=8, n_iter=80)
    recon = w @ h
    err = np.linalg.norm(binary - recon)
    assert err < np.linalg.norm(binary)            # better than predicting zeros
    assert np.all(np.diff(norms) <= 1e-9)          # component norms sorted desc


def test_image_feature_matrix_shape_and_binary_channel():
    rng = np.random.default_rng(2)
    imgs = np.zeros((3, *IMAGE_SHAPE))
    for img in imgs:
        paint_single_bit_scatter(img, rng)
    paint_vertical_line(imgs[0], col=23, grade=6.0)
    feats = image_feature_matrix(imgs, pool=2, k_eigen=16)
    expected = 2 * (128 // 2) * (46 // 2) + 16
    assert feats.shape == (3, expected)
    assert np.isfinite(feats).all()


def test_single_bit_scatter_stays_below_binarization_threshold():
    rng = np.random.default_rng(3)
    img = np.zeros(IMAGE_SHAPE)
    paint_single_bit_scatter(img, rng, density=0.05, grade_range=(1, 2))
    # random single-bit fails are low grade -> removed by the grade>=3 binarization
    assert binarize_fbm(img).sum() == 0
