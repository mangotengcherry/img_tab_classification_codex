"""Synthetic FBM + electrical dataset that embodies the modality asymmetry.

There is no real data in this repo, so this generator produces a controllable
stand-in whose structure mirrors the real problem, letting the whole pipeline
(train -> predict -> evaluate -> visualize) run end-to-end. Teammates replace it
with real chips (see docs/fusion_eval_quickstart.md).

Built-in structure:

- Image is a 128x46 grade-0..8 FBM.
- Two *spatial* classes are separable in the image (``edge_ring``, ``center_blob``).
- Two *identity* classes look IDENTICAL in the image (same faint vertical stripe)
  and are separable ONLY by electrical features (``leak_top`` vs ``leak_bottom``)
  — this is the case that synthetic images can never teach.
- Electrical (tabular) features are organized into top/middle/bottom regions; the
  identity classes light up different regions.
- ``real_single`` / ``real_composite`` rows carry tabular; ``synthetic_composite``
  rows are composed from two single-defect images (np.maximum) and carry NO
  tabular (tabular cannot be synthesized).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


IMAGE_SHAPE = (128, 46)
LABELS = ["edge_ring", "center_blob", "leak_top", "leak_bottom"]
IDENTITY_LABELS = ["leak_top", "leak_bottom"]
N_TABULAR = 201  # MSR_000 .. MSR_200
MAX_GRADE = 8.0

# tabular region slices (in reality this comes from measurement_map.csv)
_TOP = slice(0, 67)
_MIDDLE = slice(67, 134)
_BOTTOM = slice(134, 201)


@dataclass
class FusionDataset:
    images: np.ndarray          # (N, 128, 46) float grades 0..8
    tabular: np.ndarray         # (N, 201) float, NaN rows for synthetic
    labels: np.ndarray          # (N, 4) int 0/1
    eval_group: np.ndarray      # (N,) str
    chip_id: list[str]
    split: np.ndarray           # (N,) str  'train' | 'test'
    label_names: list[str]
    identity_labels: list[str]

    @property
    def has_tabular(self) -> np.ndarray:
        return ~np.isnan(self.tabular).all(axis=1)

    @property
    def images_flat(self) -> np.ndarray:
        return self.images.reshape(self.images.shape[0], -1)


def _image_for(active: list[int], rng: np.random.Generator) -> np.ndarray:
    h, w = IMAGE_SHAPE
    img = rng.uniform(0.0, 0.4, size=(h, w))  # faint background noise
    for k in active:
        name = LABELS[k]
        if name == "edge_ring":
            img[:6, :] += 6.0
            img[-6:, :] += 6.0
            img[:, :3] += 6.0
            img[:, -3:] += 6.0
        elif name == "center_blob":
            cy, cx = h // 2, w // 2
            img[cy - 12 : cy + 12, cx - 8 : cx + 8] += 6.5
        elif name in ("leak_top", "leak_bottom"):
            # IDENTICAL image signature for both identity classes:
            # a faint vertical stripe in the middle columns. The image cannot
            # tell top from bottom; only the electrical features can.
            img[:, w // 2 - 2 : w // 2 + 2] += 3.0
    img += rng.normal(0.0, 0.3, size=(h, w))
    return np.clip(img, 0.0, MAX_GRADE)


def _tabular_for(active: list[int], rng: np.random.Generator) -> np.ndarray:
    feats = rng.normal(0.5, 0.3, size=N_TABULAR)  # baseline leakage floor
    for k in active:
        name = LABELS[k]
        if name == "edge_ring":
            idx = rng.choice(N_TABULAR, size=20, replace=False)
            feats[idx] += rng.uniform(2.0, 4.0, size=20)
        elif name == "center_blob":
            feats[_MIDDLE] += rng.uniform(1.5, 3.0, size=(_MIDDLE.stop - _MIDDLE.start))
        elif name == "leak_top":
            feats[_TOP] += rng.uniform(3.0, 6.0, size=(_TOP.stop - _TOP.start))
        elif name == "leak_bottom":
            feats[_BOTTOM] += rng.uniform(3.0, 6.0, size=(_BOTTOM.stop - _BOTTOM.start))
    return np.clip(feats, 0.0, None)


def _compose_images(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(np.maximum(a, b), 0.0, MAX_GRADE)


def generate_dataset(
    *,
    seed: int = 0,
    n_real_single_train: int = 520,
    n_real_composite_train: int = 70,
    n_synth_composite_train: int = 460,
    n_real_single_test: int = 180,
    n_real_composite_test: int = 60,
    n_synth_composite_test: int = 150,
) -> FusionDataset:
    """Generate a train/test FBM fusion dataset with the modality asymmetry."""
    rng = np.random.default_rng(seed)
    n_lab = len(LABELS)

    images: list[np.ndarray] = []
    tabular: list[np.ndarray] = []
    labels: list[list[int]] = []
    groups: list[str] = []
    splits: list[str] = []
    chip_ids: list[str] = []
    counter = 0

    def add_single(split: str) -> tuple[np.ndarray, list[int]]:
        nonlocal counter
        k = int(rng.integers(0, n_lab))
        active = [k]
        img = _image_for(active, rng)
        tab = _tabular_for(active, rng)
        y = [1 if i in active else 0 for i in range(n_lab)]
        images.append(img)
        tabular.append(tab)
        labels.append(y)
        groups.append("real_single")
        splits.append(split)
        chip_ids.append(f"R{counter:05d}")
        counter += 1
        return img, active

    def add_real_composite(split: str) -> None:
        nonlocal counter
        pair = list(rng.choice(n_lab, size=2, replace=False))
        img = _image_for(pair, rng)
        tab = _tabular_for(pair, rng)
        y = [1 if i in pair else 0 for i in range(n_lab)]
        images.append(img)
        tabular.append(tab)
        labels.append(y)
        groups.append("real_composite")
        splits.append(split)
        chip_ids.append(f"R{counter:05d}")
        counter += 1

    def add_synth_composite(split: str) -> None:
        nonlocal counter
        pair = list(rng.choice(n_lab, size=2, replace=False))
        img_a = _image_for([pair[0]], rng)
        img_b = _image_for([pair[1]], rng)
        img = _compose_images(img_a, img_b)
        y = [1 if i in pair else 0 for i in range(n_lab)]
        images.append(img)
        tabular.append(np.full(N_TABULAR, np.nan))  # tabular cannot be synthesized
        labels.append(y)
        groups.append("synthetic_composite")
        splits.append(split)
        chip_ids.append(f"S{counter:05d}")
        counter += 1

    for _ in range(n_real_single_train):
        add_single("train")
    for _ in range(n_real_composite_train):
        add_real_composite("train")
    for _ in range(n_synth_composite_train):
        add_synth_composite("train")
    for _ in range(n_real_single_test):
        add_single("test")
    for _ in range(n_real_composite_test):
        add_real_composite("test")
    for _ in range(n_synth_composite_test):
        add_synth_composite("test")

    return FusionDataset(
        images=np.asarray(images, dtype=float),
        tabular=np.asarray(tabular, dtype=float),
        labels=np.asarray(labels, dtype=int),
        eval_group=np.asarray(groups, dtype=object),
        chip_id=chip_ids,
        split=np.asarray(splits, dtype=object),
        label_names=list(LABELS),
        identity_labels=list(IDENTITY_LABELS),
    )
