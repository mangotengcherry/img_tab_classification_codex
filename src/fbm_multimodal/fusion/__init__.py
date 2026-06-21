"""Multi-modal fusion evaluation and diagnostics.

This subpackage is intentionally self-contained: it does NOT import from the
core ``fbm_multimodal`` modules (``metrics``, ``condition_eval`` …) so that it
never breaks while those files are under active, concurrent development.

It evaluates the three model heads described in ``plan.md`` (image-only,
tabular-only, fusion) and adds fusion-specific diagnostics that the core
condition evaluator does not cover:

- per-head subset accuracy stratified by ``eval_group``
- KPI product (single x composite) per head
- fusion gain over the best single-modality head
- modality-collapse diagnostic (does fusion ignore the tabular branch?)
- identity-class slice (classes separable only by electrical features)

See ``docs/multimodal_fusion_guide.md`` and ``docs/fusion_eval_quickstart.md``.
"""

from fbm_multimodal.fusion.fusion_eval import (
    FusionEvalReport,
    HEAD_PROB_PREFIXES,
    evaluate_fusion,
    modality_contribution,
    wilson_ci,
)

__all__ = [
    "FusionEvalReport",
    "HEAD_PROB_PREFIXES",
    "evaluate_fusion",
    "modality_contribution",
    "wilson_ci",
]
