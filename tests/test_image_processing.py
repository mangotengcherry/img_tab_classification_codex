import numpy as np
import pytest

from fbm_multimodal.image import (
    FBMImageAugmenter,
    normalize_fbm_intensity,
)


def test_normalize_fbm_intensity_scales_zero_to_eight_grades():
    image = np.array([[0, 4, 8]], dtype=np.uint8)

    normalized = normalize_fbm_intensity(image)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, np.array([[0.0, 0.5, 1.0]], dtype=np.float32))


def test_normalize_fbm_intensity_rejects_values_outside_grade_range():
    image = np.array([[0, 9]], dtype=np.uint8)

    with pytest.raises(ValueError, match="0..8"):
        normalize_fbm_intensity(image)


def test_vertical_flip_requires_physical_metadata_alignment():
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    augmenter = FBMImageAugmenter(horizontal_flip=True, vertical_flip=True)

    with pytest.raises(ValueError, match="physical metadata"):
        augmenter.apply(image, allow_vertical=False)


def test_vertical_flip_updates_top_bottom_regions_when_allowed():
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    augmenter = FBMImageAugmenter(horizontal_flip=False, vertical_flip=True)

    flipped, updated_regions = augmenter.apply(
        image,
        allow_vertical=True,
        physical_regions=["top", "middle", "bottom", "unknown"],
    )

    np.testing.assert_array_equal(flipped, np.flipud(image))
    assert updated_regions == ["bottom", "middle", "top", "unknown"]
