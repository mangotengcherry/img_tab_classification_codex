"""Evaluation and diagnostics for image+tabular fusion under modality asymmetry.

Context
-------
In this project images can be synthesized/augmented but tabular (electrical MSR)
features cannot. So:

- ``synthetic_composite`` samples have an image but NO tabular -> only the
  image head is defined for them.
- The fusion head can only be trained/evaluated where image AND tabular AND
  label co-exist, i.e. on REAL samples (and real composite is scarce).
- The "identity" classes (visually identical, separable only electrically)
  cannot be learned from synthetic images at all -> there the tabular/fusion
  heads must carry the signal.

This module evaluates that situation honestly from a *predictions table* (no
model object required) and flags the two classic failure modes:

1. fusion collapsing toward the image modality (ignoring tabular), and
2. the identity-class slice where tabular must dominate.

Predictions table contract
---------------------------
A ``pandas.DataFrame`` (or CSV) with, per chip row:

- ``eval_group``: one of ``real_single`` / ``real_composite`` /
  ``synthetic_composite`` (configurable).
- ``true_<label>`` in {0, 1} for every label.
- ``image_prob_<label>`` in [0, 1] for every label.
- ``tabular_prob_<label>`` in [0, 1] for every label (NaN where unavailable,
  e.g. synthetic rows).
- ``fusion_prob_<label>`` in [0, 1] for every label (NaN where unavailable).
  ``prob_<label>`` is accepted as an alias for the fusion head so the SAME CSV
  also works with the core ``evaluate-conditions`` CLI.
- optional ``chip_id`` and a grouping column (e.g. ``wafer_id``) for grouped
  reporting.

A head is considered "available" for a row only if all of its probability
columns are present and non-NaN on that row. This is what encodes the modality
asymmetry: the tabular/fusion heads are automatically skipped on synthetic
rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd


# Head name -> the probability-column prefix it reads. The fusion head also
# accepts the bare ``prob_`` prefix as an alias (see _head_columns).
HEAD_PROB_PREFIXES: dict[str, str] = {
    "image_only": "image_prob_",
    "tabular_only": "tabular_prob_",
    "fusion": "fusion_prob_",
}

DEFAULT_SINGLE_GROUP = "real_single"
DEFAULT_COMPOSITE_GROUP = "real_composite"
DEFAULT_SYNTHETIC_GROUP = "synthetic_composite"
DEFAULT_REAL_ALL_GROUP = "real_all"


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (0.0, 1.0) for an empty sample so that tiny supports are visibly
    uninformative rather than silently confident.
    """
    if total <= 0:
        return (0.0, 1.0)
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    margin = (z * math.sqrt(phat * (1.0 - phat) / total + z * z / (4.0 * total * total))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True)
class SubsetResult:
    """Subset-accuracy of one head on one slice of rows."""

    accuracy: float
    support: int
    correct: int
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "support": float(self.support),
            "correct": float(self.correct),
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


@dataclass
class FusionEvalReport:
    """Structured fusion evaluation result."""

    labels: list[str]
    thresholds: dict[str, float]
    heads_present: list[str]
    # head -> eval_group -> SubsetResult
    head_group_accuracy: dict[str, dict[str, SubsetResult]]
    # head -> {single_acc, composite_acc, kpi_product, single_support, composite_support}
    kpi_by_head: dict[str, dict[str, float]]
    fusion_gain: dict[str, float]
    collapse_diagnostic: dict[str, float]
    identity_slice: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "labels": self.labels,
            "thresholds": self.thresholds,
            "heads_present": self.heads_present,
            "head_group_accuracy": {
                head: {group: res.to_dict() for group, res in groups.items()}
                for head, groups in self.head_group_accuracy.items()
            },
            "kpi_by_head": self.kpi_by_head,
            "fusion_gain": self.fusion_gain,
            "collapse_diagnostic": self.collapse_diagnostic,
            "identity_slice": self.identity_slice,
            "warnings": self.warnings,
        }

    def render(self) -> str:
        """Human-readable markdown summary."""
        return _render_report(self)


def evaluate_fusion(
    predictions: pd.DataFrame,
    *,
    labels: Sequence[str],
    thresholds: dict[str, float] | float = 0.5,
    group_column: str = "eval_group",
    single_group: str = DEFAULT_SINGLE_GROUP,
    composite_group: str = DEFAULT_COMPOSITE_GROUP,
    synthetic_group: str = DEFAULT_SYNTHETIC_GROUP,
    identity_labels: Sequence[str] | None = None,
    min_support_warn: int = 30,
    collapse_min_candidates: int = 5,
    collapse_follow_rate_warn: float = 0.5,
) -> FusionEvalReport:
    """Evaluate image-only / tabular-only / fusion heads and fusion diagnostics.

    Parameters mirror the predictions-table contract documented at the top of
    this module. ``identity_labels`` are the labels whose classes are claimed to
    be separable only by electrical features; the identity slice reports whether
    the tabular/fusion heads actually beat the image head there.
    """
    labels = list(labels)
    if not labels:
        raise ValueError("at least one label is required")
    if group_column not in predictions.columns:
        raise ValueError(f"predictions must contain the group column '{group_column}'")
    _require_true_columns(predictions, labels)

    threshold_map = _resolve_thresholds(labels, thresholds)
    heads_present = [head for head in HEAD_PROB_PREFIXES if _head_columns(predictions, head, labels) is not None]
    if not heads_present:
        raise ValueError(
            "no head probability columns found; expected at least one of "
            "image_prob_<label>, tabular_prob_<label>, fusion_prob_<label> (or prob_<label>)"
        )

    true = _true_matrix(predictions, labels)
    # head -> (pred matrix int, available mask) aligned to predictions index order
    head_pred: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for head in heads_present:
        head_pred[head] = _binarize_head(predictions, head, labels, threshold_map)

    groups = predictions[group_column].astype(str).to_numpy()
    warnings: list[str] = []

    # ---- per head x per group subset accuracy ----------------------------
    all_groups = [single_group, composite_group, DEFAULT_REAL_ALL_GROUP, synthetic_group]
    present_groups = [g for g in all_groups if (groups == g).any()]
    if (groups == single_group).any() or (groups == composite_group).any():
        present_groups = [g for g in present_groups if g != DEFAULT_REAL_ALL_GROUP]
        insert_at = 0
        if single_group in present_groups:
            insert_at = present_groups.index(single_group) + 1
        if composite_group in present_groups:
            insert_at = present_groups.index(composite_group) + 1
        present_groups.insert(insert_at, DEFAULT_REAL_ALL_GROUP)
    head_group_accuracy: dict[str, dict[str, SubsetResult]] = {}
    for head in heads_present:
        pred, avail = head_pred[head]
        per_group: dict[str, SubsetResult] = {}
        for group in present_groups:
            if group == DEFAULT_REAL_ALL_GROUP:
                mask = np.isin(groups, np.array([single_group, composite_group], dtype=object)) & avail
            else:
                mask = (groups == group) & avail
            per_group[group] = _subset_result(true[mask], pred[mask])
        head_group_accuracy[head] = per_group

    # ---- KPI product per head (single x composite, real only) ------------
    kpi_by_head: dict[str, dict[str, float]] = {}
    for head in heads_present:
        single_res = head_group_accuracy[head].get(single_group)
        composite_res = head_group_accuracy[head].get(composite_group)
        single_acc = single_res.accuracy if single_res else float("nan")
        composite_acc = composite_res.accuracy if composite_res else float("nan")
        kpi_by_head[head] = {
            "single_acc": single_acc,
            "composite_acc": composite_acc,
            "kpi_product": single_acc * composite_acc,
            "single_support": float(single_res.support if single_res else 0),
            "composite_support": float(composite_res.support if composite_res else 0),
            "composite_ci_low": composite_res.ci_low if composite_res else float("nan"),
            "composite_ci_high": composite_res.ci_high if composite_res else float("nan"),
        }
        if composite_res and 0 < composite_res.support < min_support_warn:
            warnings.append(
                f"[{head}] composite support is only {composite_res.support} (< {min_support_warn}); "
                f"composite_acc CI=[{composite_res.ci_low:.2f}, {composite_res.ci_high:.2f}] is too wide for a decision."
            )

    # ---- fusion gain over best single modality ---------------------------
    fusion_gain: dict[str, float] = {}
    if "fusion" in heads_present:
        unimodal = [h for h in ("image_only", "tabular_only") if h in heads_present]
        best_unimodal_kpi = max((kpi_by_head[h]["kpi_product"] for h in unimodal), default=float("nan"))
        fusion_gain["fusion_kpi"] = kpi_by_head["fusion"]["kpi_product"]
        fusion_gain["best_unimodal_kpi"] = best_unimodal_kpi
        fusion_gain["gain"] = kpi_by_head["fusion"]["kpi_product"] - best_unimodal_kpi
        for h in unimodal:
            fusion_gain[f"gain_over_{h}"] = (
                kpi_by_head["fusion"]["kpi_product"] - kpi_by_head[h]["kpi_product"]
            )

    # ---- modality-collapse diagnostic on the composite-real group --------
    collapse_diagnostic = _collapse_diagnostic(
        true,
        head_pred,
        groups == composite_group,
        heads_present,
        min_candidates=collapse_min_candidates,
        follow_rate_warn=collapse_follow_rate_warn,
        warnings=warnings,
    )

    # ---- identity-class slice -------------------------------------------
    identity_slice: dict[str, object] = {}
    if identity_labels:
        identity_slice = _identity_slice(
            predictions,
            true,
            head_pred,
            labels,
            list(identity_labels),
            groups,
            real_groups=(single_group, composite_group),
            heads_present=heads_present,
            warnings=warnings,
        )

    return FusionEvalReport(
        labels=labels,
        thresholds=threshold_map,
        heads_present=heads_present,
        head_group_accuracy=head_group_accuracy,
        kpi_by_head=kpi_by_head,
        fusion_gain=fusion_gain,
        collapse_diagnostic=collapse_diagnostic,
        identity_slice=identity_slice,
        warnings=warnings,
    )


def modality_contribution(
    predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    images: np.ndarray,
    tabular: np.ndarray,
    y_true: np.ndarray,
    *,
    thresholds: np.ndarray | float = 0.5,
    null_tabular: np.ndarray | None = None,
) -> dict[str, float]:
    """Model-side, framework-agnostic true ablation of the tabular modality.

    ``predict_fn(images, tabular) -> probabilities`` of shape (n, n_labels).
    Runs the model with the real tabular input and again with a null tabular
    input, and reports the subset-accuracy drop. A drop near 0 means the fusion
    model is ignoring the tabular branch (image collapse).

    This is the only function that needs the actual model; everything else works
    from a predictions table.
    """
    images = np.asarray(images)
    tabular = np.asarray(tabular, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    if null_tabular is None:
        null_tabular = np.zeros_like(tabular)

    base_prob = np.asarray(predict_fn(images, tabular), dtype=float)
    ablated_prob = np.asarray(predict_fn(images, null_tabular), dtype=float)
    thr = thresholds if np.ndim(thresholds) else float(thresholds)

    base_pred = (base_prob >= thr).astype(int)
    ablated_pred = (ablated_prob >= thr).astype(int)

    base_acc = float((base_pred == y_true).all(axis=1).mean()) if len(y_true) else float("nan")
    ablated_acc = float((ablated_pred == y_true).all(axis=1).mean()) if len(y_true) else float("nan")
    return {
        "subset_acc_with_tabular": base_acc,
        "subset_acc_tabular_ablated": ablated_acc,
        "tabular_contribution": base_acc - ablated_acc,
        "n": float(len(y_true)),
    }


def run_leakage_checks(
    predictions: pd.DataFrame,
    *,
    tensorizer_fit_sample_ids: set[str] | None = None,
    train_real_sample_ids: set[str] | None = None,
    catboost_metadata: dict[str, object] | None = None,
    pseudo_labeling_enabled: bool = False,
    group_column: str = "eval_group",
    synthetic_column: str = "is_synthetic",
    official_groups: Sequence[str] = (DEFAULT_SINGLE_GROUP, DEFAULT_COMPOSITE_GROUP),
) -> list[str]:
    """Return warnings for common leakage risks in WL/CatBoost fusion runs."""
    warnings_out: list[str] = []

    if tensorizer_fit_sample_ids is not None and train_real_sample_ids is not None:
        leaked = set(tensorizer_fit_sample_ids) - set(train_real_sample_ids)
        if leaked:
            preview = ", ".join(sorted(leaked)[:5])
            warnings_out.append(
                f"WL baseline fit includes {len(leaked)} samples outside train_real_sample_ids: {preview}"
            )

    metadata = catboost_metadata or {}
    train_prediction_mode = str(metadata.get("train_prediction_mode", "oof")).lower()
    if train_prediction_mode != "oof":
        warnings_out.append(
            f"CatBoost train logits must be OOF, got train_prediction_mode={train_prediction_mode!r}"
        )
    if metadata.get("synthetic_excluded") is False:
        warnings_out.append("Synthetic samples are not excluded from CatBoost training metadata")

    if pseudo_labeling_enabled:
        warnings_out.append("Pseudo-labeling is enabled; default production config should keep it disabled")

    if group_column in predictions.columns and synthetic_column in predictions.columns:
        official_mask = predictions[group_column].astype(str).isin(list(official_groups))
        synthetic_mask = predictions[synthetic_column].astype(bool)
        n_bad = int((official_mask & synthetic_mask).sum())
        if n_bad:
            warnings_out.append(
                f"{n_bad} synthetic rows are marked as official metric groups; synthetic samples must be auxiliary only"
            )

    return warnings_out


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------
def _resolve_thresholds(labels: list[str], thresholds: dict[str, float] | float) -> dict[str, float]:
    if isinstance(thresholds, dict):
        return {label: float(thresholds.get(label, 0.5)) for label in labels}
    return {label: float(thresholds) for label in labels}


def _head_columns(frame: pd.DataFrame, head: str, labels: list[str]) -> list[str] | None:
    """Return the probability columns for a head, or None if absent.

    The fusion head falls back to the bare ``prob_<label>`` alias so a single
    prediction CSV is compatible with the core condition evaluator.
    """
    prefix = HEAD_PROB_PREFIXES[head]
    columns = [f"{prefix}{label}" for label in labels]
    if all(column in frame.columns for column in columns):
        return columns
    if head == "fusion":
        alias = [f"prob_{label}" for label in labels]
        if all(column in frame.columns for column in alias):
            return alias
    return None


def _require_true_columns(frame: pd.DataFrame, labels: list[str]) -> None:
    missing = [f"true_{label}" for label in labels if f"true_{label}" not in frame.columns]
    if missing:
        raise ValueError(f"predictions is missing true label columns: {missing}")


def _true_matrix(frame: pd.DataFrame, labels: list[str]) -> np.ndarray:
    return frame[[f"true_{label}" for label in labels]].astype(int).to_numpy()


def _binarize_head(
    frame: pd.DataFrame, head: str, labels: list[str], threshold_map: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    columns = _head_columns(frame, head, labels)
    values = frame[columns].to_numpy(dtype=float)
    available = ~np.isnan(values).any(axis=1)
    thr = np.array([threshold_map[label] for label in labels], dtype=float)
    pred = np.zeros_like(values, dtype=int)
    pred[available] = (values[available] >= thr).astype(int)
    return pred, available


def _subset_result(true: np.ndarray, pred: np.ndarray) -> SubsetResult:
    n = int(true.shape[0])
    if n == 0:
        return SubsetResult(accuracy=float("nan"), support=0, correct=0, ci_low=0.0, ci_high=1.0)
    correct_mask = (true == pred).all(axis=1)
    correct = int(correct_mask.sum())
    accuracy = correct / n
    ci_low, ci_high = wilson_ci(correct, n)
    return SubsetResult(accuracy=accuracy, support=n, correct=correct, ci_low=ci_low, ci_high=ci_high)


def _row_correct(true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Boolean per-row subset-correctness over rows selected by mask."""
    if not mask.any():
        return np.zeros(0, dtype=bool)
    return (true[mask] == pred[mask]).all(axis=1)


def _collapse_diagnostic(
    true: np.ndarray,
    head_pred: dict[str, tuple[np.ndarray, np.ndarray]],
    composite_mask: np.ndarray,
    heads_present: list[str],
    *,
    min_candidates: int,
    follow_rate_warn: float,
    warnings: list[str],
) -> dict[str, float]:
    """Does fusion follow the tabular head when tabular is right and image is wrong?

    On the real-composite slice, find rows where image-only is subset-wrong but
    tabular-only is subset-right ("tabular rescue candidates"). The fraction of
    those that fusion gets right is the follow rate; a low follow rate means
    fusion is ignoring tabular and collapsing toward image.
    """
    result: dict[str, float] = {}
    needed = {"image_only", "tabular_only", "fusion"}
    if not needed.issubset(set(heads_present)):
        result["available"] = 0.0
        return result

    img_pred, img_avail = head_pred["image_only"]
    tab_pred, tab_avail = head_pred["tabular_only"]
    fus_pred, fus_avail = head_pred["fusion"]
    mask = composite_mask & img_avail & tab_avail & fus_avail

    n = int(mask.sum())
    result["available"] = 1.0
    result["n_composite"] = float(n)
    if n == 0:
        return result

    img_ok = _row_correct(true, img_pred, mask)
    tab_ok = _row_correct(true, tab_pred, mask)
    fus_ok = _row_correct(true, fus_pred, mask)

    img_acc = float(img_ok.mean())
    fus_acc = float(fus_ok.mean())
    result["image_only_subset_acc"] = img_acc
    result["fusion_subset_acc"] = fus_acc
    result["fusion_gain_over_image"] = fus_acc - img_acc

    rescue = (~img_ok) & tab_ok  # tabular knows something the image misses
    rescue_total = int(rescue.sum())
    fusion_followed = int((rescue & fus_ok).sum())
    result["tabular_rescue_candidates"] = float(rescue_total)
    result["fusion_followed_tabular"] = float(fusion_followed)
    result["fusion_follow_rate"] = (fusion_followed / rescue_total) if rescue_total else float("nan")

    regressions = img_ok & (~fus_ok)  # fusion broke something image had right
    result["fusion_regressions"] = float(int(regressions.sum()))

    if rescue_total >= min_candidates and result["fusion_follow_rate"] < follow_rate_warn:
        warnings.append(
            f"[collapse] fusion captured only {fusion_followed}/{rescue_total} "
            f"({result['fusion_follow_rate']:.0%}) of cases the tabular head rescued; "
            "fusion may be collapsing toward the image modality."
        )
    if result["fusion_gain_over_image"] <= 0:
        warnings.append(
            "[collapse] fusion is no better than image-only on real composite "
            f"(gain={result['fusion_gain_over_image']:+.3f}); the tabular branch is not contributing."
        )
    return result


def _identity_slice(
    predictions: pd.DataFrame,
    true: np.ndarray,
    head_pred: dict[str, tuple[np.ndarray, np.ndarray]],
    labels: list[str],
    identity_labels: list[str],
    groups: np.ndarray,
    *,
    real_groups: tuple[str, str],
    heads_present: list[str],
    warnings: list[str],
) -> dict[str, object]:
    """Evaluate the slice of real chips that carry an identity (electrical-only) label."""
    unknown = [label for label in identity_labels if label not in labels]
    if unknown:
        raise ValueError(f"identity_labels not in labels: {unknown}")

    identity_idx = [labels.index(label) for label in identity_labels]
    has_identity = true[:, identity_idx].any(axis=1)
    real_mask = np.isin(groups, np.array(real_groups))
    slice_mask = has_identity & real_mask

    out: dict[str, object] = {
        "identity_labels": identity_labels,
        "n": float(int(slice_mask.sum())),
        "by_head": {},
    }
    if not slice_mask.any():
        warnings.append("[identity] no real chips carry an identity label; slice is empty.")
        return out

    accuracies: dict[str, float] = {}
    for head in heads_present:
        pred, avail = head_pred[head]
        mask = slice_mask & avail
        res = _subset_result(true[mask], pred[mask])
        out["by_head"][head] = res.to_dict()  # type: ignore[index]
        accuracies[head] = res.accuracy

    if "tabular_only" in accuracies and "image_only" in accuracies:
        advantage = accuracies["tabular_only"] - accuracies["image_only"]
        out["tabular_minus_image"] = advantage
        if not math.isnan(advantage) and advantage <= 0:
            warnings.append(
                "[identity] tabular head does NOT beat image head on the identity slice "
                f"(tabular-image={advantage:+.3f}); either the identity-class hypothesis or the "
                "tabular features/labels need review."
            )
    return out


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.3f}"
    return str(value)


def _render_report(report: FusionEvalReport) -> str:
    lines: list[str] = []
    lines.append("# Fusion evaluation report")
    lines.append("")
    lines.append(f"- labels: {', '.join(report.labels)}")
    lines.append(f"- heads present: {', '.join(report.heads_present)}")
    lines.append("")

    lines.append("## Subset accuracy by head x eval_group  (acc [CI] / support)")
    lines.append("")
    groups = sorted({g for head in report.head_group_accuracy.values() for g in head})
    header = "| head | " + " | ".join(groups) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(groups) + 1))
    for head, per_group in report.head_group_accuracy.items():
        cells = []
        for group in groups:
            res = per_group.get(group)
            if res is None or res.support == 0:
                cells.append("—")
            else:
                cells.append(f"{res.accuracy:.3f} [{res.ci_low:.2f},{res.ci_high:.2f}] / {res.support}")
        lines.append(f"| {head} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## KPI product per head  (single x composite, real only)")
    lines.append("")
    lines.append("| head | single | composite | KPI product | composite support |")
    lines.append("|---|---|---|---|---|")
    for head, kpi in report.kpi_by_head.items():
        lines.append(
            f"| {head} | {_fmt(kpi['single_acc'])} | {_fmt(kpi['composite_acc'])} | "
            f"{_fmt(kpi['kpi_product'])} | {int(kpi['composite_support'])} |"
        )
    lines.append("")

    if report.fusion_gain:
        lines.append("## Fusion gain")
        lines.append("")
        for key, value in report.fusion_gain.items():
            lines.append(f"- {key}: {_fmt(value)}")
        lines.append("")

    if report.collapse_diagnostic.get("available"):
        lines.append("## Modality-collapse diagnostic (real composite)")
        lines.append("")
        for key, value in report.collapse_diagnostic.items():
            if key == "available":
                continue
            lines.append(f"- {key}: {_fmt(value)}")
        lines.append("")

    if report.identity_slice:
        lines.append("## Identity-class slice (electrical-only classes)")
        lines.append("")
        lines.append(f"- identity labels: {', '.join(report.identity_slice.get('identity_labels', []))}")
        lines.append(f"- n: {int(report.identity_slice.get('n', 0))}")
        by_head = report.identity_slice.get("by_head", {})
        for head, res in by_head.items():  # type: ignore[union-attr]
            lines.append(f"- {head}: acc={_fmt(res['accuracy'])} / support={int(res['support'])}")
        if "tabular_minus_image" in report.identity_slice:
            lines.append(f"- tabular_minus_image: {_fmt(report.identity_slice['tabular_minus_image'])}")
        lines.append("")

    if report.warnings:
        lines.append("## ⚠️ Warnings")
        lines.append("")
        for warning in report.warnings:
            lines.append(f"- {warning}")
        lines.append("")
    else:
        lines.append("_No warnings._")
        lines.append("")

    return "\n".join(lines)
