from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


REGION_VERTICAL_FLIP = {
    "top": "bottom",
    "bottom": "top",
    "middle": "middle",
    "unknown": "unknown",
}


def normalize_fbm_intensity(image: np.ndarray, max_grade: int = 8) -> np.ndarray:
    """Normalize FBM grade intensities from 0..max_grade to 0..1."""
    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError("FBM image is empty")
    min_value = float(np.nanmin(arr))
    max_value = float(np.nanmax(arr))
    if min_value < 0 or max_value > max_grade:
        raise ValueError(f"FBM intensity values must be in 0..{max_grade}; got {min_value}..{max_value}")
    return (arr.astype(np.float32) / float(max_grade)).astype(np.float32)


@dataclass(frozen=True)
class FBMImageAugmenter:
    """Deterministic FBM flip helper used by experiment ablations."""

    horizontal_flip: bool = False
    vertical_flip: bool = False

    def apply(
        self,
        image: np.ndarray,
        *,
        allow_vertical: bool = False,
        physical_regions: Iterable[str] | None = None,
    ) -> tuple[np.ndarray, list[str] | None]:
        arr = np.asarray(image)
        updated_regions = list(physical_regions) if physical_regions is not None else None

        if self.horizontal_flip:
            arr = np.fliplr(arr)

        if self.vertical_flip:
            if not allow_vertical:
                raise ValueError("Vertical flip requires aligned physical metadata")
            arr = np.flipud(arr)
            if updated_regions is not None:
                updated_regions = [REGION_VERTICAL_FLIP.get(region, "unknown") for region in updated_regions]

        return arr.copy(), updated_regions
