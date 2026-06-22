"""Matplotlib visualizations for the fusion experiment.

Headless (Agg) so it runs in CI / over SSH. Every function saves a PNG and
returns its path. ``generate_all_figures`` wires them together for the runner.
"""

from __future__ import annotations

from itertools import combinations
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fbm-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fbm_multimodal.fusion.data import FusionDataset  # noqa: E402
from fbm_multimodal.fusion.fbm_patterns import (  # noqa: E402
    IMAGE_SHAPE,
    MAX_GRADE,
    paint_cluster,
    paint_edge_ring,
    paint_horizontal_line,
    paint_single_bit_scatter,
    paint_vertical_line,
)
from fbm_multimodal.fusion.fusion_eval import FusionEvalReport  # noqa: E402


_HEAD_COLORS = {"image_only": "#4C78A8", "tabular_only": "#F58518", "fusion": "#54A24B"}
_HEAD_ORDER = ["image_only", "tabular_only", "fusion"]


def _first_index(dataset: FusionDataset, want: list[int], group: str | None = None) -> int | None:
    for i in range(dataset.labels.shape[0]):
        active = [k for k in range(dataset.labels.shape[1]) if dataset.labels[i, k] == 1]
        if active == want and (group is None or dataset.eval_group[i] == group):
            return i
    return None


def plot_dataset_overview(dataset: FusionDataset, out: Path) -> Path:
    """FBM examples (incl. the two identical-looking identity classes) + group counts."""
    names = dataset.label_names
    panels = [
        ("edge_ring", [names.index("edge_ring")]),
        ("center_blob", [names.index("center_blob")]),
        ("leak_top", [names.index("leak_top")]),
        ("leak_bottom", [names.index("leak_bottom")]),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 4.5))
    for ax, (title, want) in zip(axes[:4], panels):
        idx = _first_index(dataset, want, group="real_single")
        img = dataset.images[idx] if idx is not None else np.zeros((128, 46))
        ax.imshow(img, cmap="inferno", vmin=0, vmax=8, aspect="auto")
        flag = "\nimage same" if title in dataset.identity_labels else ""
        ax.set_title(f"{title}{flag}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = axes[4]
    groups, counts = np.unique(dataset.eval_group.astype(str), return_counts=True)
    order = ["real_single", "real_composite", "synthetic_composite"]
    counts = [int(counts[list(groups).index(g)]) if g in groups else 0 for g in order]
    bars = ax.bar(range(len(order)), counts, color=["#4C78A8", "#E45756", "#B279A2"])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([g.replace("_", "\n") for g in order], fontsize=8)
    ax.set_title("sample counts by group", fontsize=10)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "FBM dataset overview — same 0-8 color scale; leak_top and leak_bottom look the same in FBM",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.25)
    return _save(fig, out, tight=False)


def plot_pattern_gallery(dataset: FusionDataset, out: Path) -> Path:
    """Show all single patterns and every 2-label synthetic composite pair."""
    names = dataset.label_names
    singles = [([i], f"single\n{names[i]}", "real_single") for i in range(len(names))]
    pairs = [
        (list(pair), f"synthetic pair\n{names[pair[0]]} + {names[pair[1]]}", "synthetic_composite")
        for pair in combinations(range(len(names)), 2)
    ]
    panels = singles + pairs

    fig, axes = plt.subplots(3, 4, figsize=(13.5, 9.2))
    axes_flat = axes.ravel()
    for ax, (want, title, group) in zip(axes_flat, panels):
        idx = _first_index(dataset, want, group=group)
        img = dataset.images[idx] if idx is not None else np.zeros((128, 46))
        ax.imshow(img, cmap="inferno", vmin=0, vmax=8, aspect="auto")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_flat[len(panels):]:
        ax.axis("off")

    fig.suptitle("Pattern coverage gallery — singles plus every two-defect synthetic overlap", fontsize=12)
    fig.text(
        0.5,
        0.025,
        "Same 0-8 color scale. The bottom row confirms identity-style overlaps still look alike in FBM.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
    )
    fig.subplots_adjust(top=0.88, bottom=0.08, wspace=0.22, hspace=0.42)
    return _save(fig, out, tight=False)


def plot_domain_pattern_stress_gallery(out: Path, *, seed: int = 0) -> Path:
    """Show domain-inspired FBM shapes without changing the model method."""
    patterns = _domain_reference_patterns(seed=seed)
    names = list(patterns)
    fig, axes = plt.subplots(2, len(names), figsize=(15.0, 5.8))

    for col, name in enumerate(names):
        img = patterns[name]
        axes[0, col].imshow(img, cmap="inferno", vmin=0, vmax=MAX_GRADE, aspect="auto")
        axes[0, col].set_title(name.replace("_", "\n"), fontsize=9)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        high_grade = (img >= 3.0).astype(float)
        axes[1, col].imshow(high_grade, cmap="gray_r", vmin=0, vmax=1, aspect="auto")
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

    axes[0, 0].set_ylabel("raw\n0-8", fontsize=9)
    axes[1, 0].set_ylabel("grade\n>=3", fontsize=9)
    fig.suptitle("Domain pattern stress gallery", fontsize=12)
    fig.text(
        0.5,
        0.025,
        "Visualization only: random low-grade scatter, line, block, and edge shapes from FBM domain examples.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
    )
    fig.subplots_adjust(top=0.84, bottom=0.12, wspace=0.18, hspace=0.18)
    return _save(fig, out, tight=False)


def _domain_reference_patterns(*, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    h, w = IMAGE_SHAPE

    def blank() -> np.ndarray:
        return rng.uniform(0.0, 0.35, size=IMAGE_SHAPE)

    random_scatter = blank()
    paint_single_bit_scatter(random_scatter, rng, density=0.035, grade_range=(1, 2))

    vertical_line = blank()
    paint_vertical_line(vertical_line, col=w // 2, width=3, grade=6.0)
    paint_single_bit_scatter(vertical_line, rng, density=0.006, grade_range=(1, 2))

    short_horizontal = blank()
    paint_horizontal_line(short_horizontal, rows=slice(h // 3, None), height=3, grade=6.0)
    paint_single_bit_scatter(short_horizontal, rng, density=0.006, grade_range=(1, 2))

    local_block = blank()
    paint_cluster(local_block, center=(h // 2, w // 2), half=(12, 8), grade=6.5)
    paint_single_bit_scatter(local_block, rng, density=0.006, grade_range=(1, 2))

    edge_ring = blank()
    paint_edge_ring(edge_ring, width=5, grade=6.0)
    paint_single_bit_scatter(edge_ring, rng, density=0.006, grade_range=(1, 2))

    return {
        "random_scatter": np.clip(random_scatter, 0.0, MAX_GRADE),
        "vertical_line": np.clip(vertical_line, 0.0, MAX_GRADE),
        "short_horizontal": np.clip(short_horizontal, 0.0, MAX_GRADE),
        "local_block": np.clip(local_block, 0.0, MAX_GRADE),
        "edge_ring": np.clip(edge_ring, 0.0, MAX_GRADE),
    }


def plot_training_curves(history: dict[str, list[float]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    styles = {
        "loss_image": ("#4C78A8", "image head (synthetic + real)"),
        "loss_tabular": ("#F58518", "tabular head (real only)"),
        "loss_fusion": ("#54A24B", "fusion head (real only)"),
    }
    for key, (color, label) in styles.items():
        if history.get(key):
            ax.plot(history[key], color=color, label=label, linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Per-head training loss (loss masking + modality dropout)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out)


def plot_head_group_accuracy(report: FusionEvalReport, out: Path) -> Path:
    groups = ["real_single", "real_composite", "synthetic_composite"]
    heads = [h for h in _HEAD_ORDER if h in report.head_group_accuracy]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    width = 0.25
    x = np.arange(len(groups))
    for j, head in enumerate(heads):
        accs, lo, hi = [], [], []
        for g in groups:
            res = report.head_group_accuracy.get(head, {}).get(g)
            if res is None or res.support == 0:
                accs.append(np.nan); lo.append(0); hi.append(0)
            else:
                accs.append(res.accuracy)
                lo.append(res.accuracy - res.ci_low)
                hi.append(res.ci_high - res.accuracy)
        offset = (j - (len(heads) - 1) / 2) * width
        ax.bar(x + offset, accs, width, label=head, color=_HEAD_COLORS.get(head),
               yerr=[lo, hi], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", "\n") for g in groups])
    ax.set_ylabel("subset accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Subset accuracy by head × eval_group  (error bars = Wilson 95% CI)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.99, 0.02, "tabular/fusion absent on synthetic (no tabular)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="gray")
    return _save(fig, out)


def plot_kpi(report: FusionEvalReport, out: Path) -> Path:
    heads = [h for h in _HEAD_ORDER if h in report.kpi_by_head]
    metrics = ["single_acc", "composite_acc", "kpi_product"]
    labels = ["single acc", "composite acc", "KPI product\n(single × composite)"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    width = 0.25
    x = np.arange(len(metrics))
    for j, head in enumerate(heads):
        vals = [report.kpi_by_head[head][m] for m in metrics]
        offset = (j - (len(heads) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=head, color=_HEAD_COLORS.get(head))
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_title("KPI product per head — fusion should beat the best single modality")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)


def plot_identity_and_collapse(report: FusionEvalReport, ablation: dict | None, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # left: identity-class slice
    ax = axes[0]
    by_head = report.identity_slice.get("by_head", {}) if report.identity_slice else {}
    heads = [h for h in _HEAD_ORDER if h in by_head]
    accs = [by_head[h]["accuracy"] for h in heads]
    bars = ax.bar(heads, accs, color=[_HEAD_COLORS.get(h) for h in heads])
    for b, v in zip(bars, accs):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("subset accuracy")
    adv = report.identity_slice.get("tabular_minus_image") if report.identity_slice else None
    sub = f"tabular − image = {adv:+.2f}" if adv is not None else ""
    ax.set_title(f"Identity classes (electrical-only)\n{sub}")
    ax.grid(axis="y", alpha=0.3)

    # right: modality-collapse diagnostic on real composite
    ax = axes[1]
    diag = report.collapse_diagnostic
    items, vals, colors = [], [], []
    if diag.get("available"):
        items += ["image\nsubset acc", "fusion\nsubset acc"]
        vals += [diag.get("image_only_subset_acc", np.nan), diag.get("fusion_subset_acc", np.nan)]
        colors += [_HEAD_COLORS["image_only"], _HEAD_COLORS["fusion"]]
        if not np.isnan(diag.get("fusion_follow_rate", np.nan)):
            items.append("fusion follows\ntabular rescue")
            vals.append(diag["fusion_follow_rate"])
            colors.append("#72B7B2")
    if ablation is not None:
        items.append("tabular\ncontribution")
        vals.append(ablation.get("tabular_contribution", np.nan))
        colors.append("#B279A2")
    bars = ax.bar(range(len(items)), vals, color=colors)
    for b, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(items, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("rate / accuracy")
    ax.set_title("Modality-collapse diagnostic (real composite)\nlow follow-rate / low contribution ⇒ collapse")
    ax.grid(axis="y", alpha=0.3)

    return _save(fig, out)


def generate_all_figures(
    dataset: FusionDataset,
    history: dict[str, list[float]],
    report: FusionEvalReport,
    out_dir: Path,
    ablation: dict | None = None,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_dataset_overview(dataset, out_dir / "01_dataset_overview.png"),
        plot_pattern_gallery(dataset, out_dir / "06_pattern_gallery.png"),
        plot_domain_pattern_stress_gallery(out_dir / "07_domain_pattern_stress_gallery.png"),
        plot_training_curves(history, out_dir / "02_training_curves.png"),
        plot_head_group_accuracy(report, out_dir / "03_head_group_accuracy.png"),
        plot_kpi(report, out_dir / "04_kpi_product.png"),
        plot_identity_and_collapse(report, ablation, out_dir / "05_identity_and_collapse.png"),
    ]


def _save(fig, out: Path, *, tight: bool = True) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out
