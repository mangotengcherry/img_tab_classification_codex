import numpy as np

from fbm_multimodal.fusion.fbm_patterns import (
    IMAGE_SHAPE,
    paint_cluster,
    paint_edge_ring,
    paint_horizontal_line,
    paint_single_bit_scatter,
    paint_vertical_line,
)


def test_vertical_line_is_a_high_grade_column():
    img = np.zeros(IMAGE_SHAPE)
    paint_vertical_line(img, col=23, width=2, grade=6.0)
    assert img[:, 23].min() >= 6.0          # the column is high grade everywhere
    assert img[:, 0].max() == 0.0           # other columns untouched


def test_horizontal_line_is_a_high_grade_row_band():
    img = np.zeros(IMAGE_SHAPE)
    paint_horizontal_line(img, rows=slice(10, None), height=3, grade=6.0)
    assert img[10:13, :].min() >= 6.0
    assert img[20, :].max() == 0.0


def test_cluster_is_a_localized_high_grade_block():
    img = np.zeros(IMAGE_SHAPE)
    paint_cluster(img, center=(64, 23), half=(10, 6), grade=6.5)
    assert img[64, 23] >= 6.5
    assert img[0, 0] == 0.0


def test_edge_ring_lights_up_borders_only():
    img = np.zeros(IMAGE_SHAPE)
    paint_edge_ring(img, width=5, grade=6.0)
    assert img[0, :].min() >= 6.0
    assert img[-1, :].min() >= 6.0
    assert img[64, 23] == 0.0               # interior untouched


def test_single_bit_scatter_is_random_and_low_grade():
    rng = np.random.default_rng(0)
    img = np.zeros(IMAGE_SHAPE)
    paint_single_bit_scatter(img, rng, density=0.02, grade_range=(1, 2))
    assert (img > 0).sum() > 0              # some fails were added
    assert img.max() < 3.0                  # but they stay low grade
