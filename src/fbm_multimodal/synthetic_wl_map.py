"""Synthetic WL residual map composition utilities."""

from __future__ import annotations

import numpy as np

from fbm_multimodal.wl_residual_map import DEFAULT_WL_CHANNELS, VALUE_CHANNELS


def compose_synthetic_wl_map(
    parent_a_map: np.ndarray,
    parent_b_map: np.ndarray,
    *,
    channel_spec: list[str] | tuple[str, ...] = DEFAULT_WL_CHANNELS,
) -> np.ndarray:
    """Compose two parent residual maps with max residual and union mask.

    Value channels use ``max(parent_a, parent_b)``. ``observed_mask`` is the
    union of parent observed cells, and ``source_count_norm`` records whether
    one or both parents supplied a cell as ``0.5`` or ``1.0``.
    """
    a = np.asarray(parent_a_map, dtype=float)
    b = np.asarray(parent_b_map, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"parent WL maps must have the same shape, got {a.shape} and {b.shape}")
    if a.ndim != 3:
        raise ValueError(f"parent WL maps must be [C, B, T], got shape {a.shape}")
    if len(channel_spec) != a.shape[0]:
        raise ValueError("channel_spec length must match the first map dimension")

    channels = {name: idx for idx, name in enumerate(channel_spec)}
    if "observed_mask" not in channels:
        raise ValueError("channel_spec must contain observed_mask")

    observed_a = a[channels["observed_mask"]] > 0
    observed_b = b[channels["observed_mask"]] > 0
    observed = observed_a | observed_b

    out = np.zeros_like(a, dtype=np.float32)
    for name, idx in channels.items():
        if name in VALUE_CHANNELS or name == "count_ratio":
            out[idx] = np.maximum(a[idx], b[idx])
            out[idx][~observed] = 0.0
    out[channels["observed_mask"]] = observed.astype(np.float32)
    if "source_count_norm" in channels:
        out[channels["source_count_norm"]] = ((observed_a.astype(float) + observed_b.astype(float)) / 2.0).astype(
            np.float32
        )
    return out
