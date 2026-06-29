"""Pseudo-labeling scaffolds.

The default project pipeline keeps pseudo-labeling disabled; modules here are
standalone utilities for future explicit opt-in experiments.
"""

from fbm_multimodal.pseudo_labeling.pairwise_topk import select_pairwise_topk

__all__ = ["select_pairwise_topk"]
