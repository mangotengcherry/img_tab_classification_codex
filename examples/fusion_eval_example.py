"""Runnable example for fusion evaluation.

Run it as-is to see a report on a tiny built-in demo dataset:

    PYTHONPATH=src python3 examples/fusion_eval_example.py

To use it for real, you only change ONE thing: replace ``build_demo_predictions()``
with a load of your own predictions CSV (see the >>> REPLACE block below). The
required columns are documented in docs/fusion_eval_quickstart.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fbm_multimodal.fusion.fusion_eval import evaluate_fusion


# Labels and which of them are "identity" classes (separable only electrically).
LABELS = ["short", "leak_top", "leak_bottom"]
IDENTITY_LABELS = ["leak_top", "leak_bottom"]  # look identical in the image


def build_demo_predictions() -> pd.DataFrame:
    """A small, deterministic demo table illustrating the modality asymmetry.

    - real_single / real_composite rows have image + tabular + fusion heads.
    - synthetic_composite rows have ONLY the image head (tabular/fusion = NaN),
      because tabular cannot be synthesized.
    - the image head cannot tell leak_top from leak_bottom (identical image);
      the tabular head can. The demo fusion head correctly follows tabular.
    """
    rng = np.random.default_rng(0)
    rows: list[dict] = []

    def img_prob(truth: int, sep: bool) -> float:
        # If the image can separate this label, it is informative; otherwise ~0.5.
        if not sep:
            return float(rng.uniform(0.4, 0.6))
        return float(rng.uniform(0.7, 0.95)) if truth else float(rng.uniform(0.05, 0.3))

    def tab_prob(truth: int) -> float:
        return float(rng.uniform(0.75, 0.97)) if truth else float(rng.uniform(0.03, 0.25))

    # real_single: one label active at a time
    for _ in range(60):
        active = rng.integers(0, len(LABELS))
        truth = [1 if i == active else 0 for i in range(len(LABELS))]
        row = {"eval_group": "real_single"}
        for i, label in enumerate(LABELS):
            separable_in_image = label not in IDENTITY_LABELS
            row[f"true_{label}"] = truth[i]
            row[f"image_prob_{label}"] = img_prob(truth[i], separable_in_image)
            row[f"tabular_prob_{label}"] = tab_prob(truth[i])
            # demo fusion: trust tabular when present
            row[f"fusion_prob_{label}"] = 0.7 * row[f"tabular_prob_{label}"] + 0.3 * row[f"image_prob_{label}"]
        rows.append(row)

    # real_composite: two labels active (scarce in reality)
    for _ in range(20):
        pair = rng.choice(len(LABELS), size=2, replace=False)
        truth = [1 if i in pair else 0 for i in range(len(LABELS))]
        row = {"eval_group": "real_composite"}
        for i, label in enumerate(LABELS):
            separable_in_image = label not in IDENTITY_LABELS
            row[f"true_{label}"] = truth[i]
            row[f"image_prob_{label}"] = img_prob(truth[i], separable_in_image)
            row[f"tabular_prob_{label}"] = tab_prob(truth[i])
            row[f"fusion_prob_{label}"] = 0.7 * row[f"tabular_prob_{label}"] + 0.3 * row[f"image_prob_{label}"]
        rows.append(row)

    # synthetic_composite: image only, tabular/fusion absent
    for _ in range(40):
        pair = rng.choice(len(LABELS), size=2, replace=False)
        truth = [1 if i in pair else 0 for i in range(len(LABELS))]
        row = {"eval_group": "synthetic_composite"}
        for i, label in enumerate(LABELS):
            separable_in_image = label not in IDENTITY_LABELS
            row[f"true_{label}"] = truth[i]
            row[f"image_prob_{label}"] = img_prob(truth[i], separable_in_image)
            row[f"tabular_prob_{label}"] = np.nan
            row[f"fusion_prob_{label}"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    # ----------------------------------------------------------------------
    # >>> REPLACE THIS BLOCK WITH YOUR REAL DATA <<<
    # predictions = pd.read_csv("outputs/fusion_predictions.csv")
    predictions = build_demo_predictions()
    # ----------------------------------------------------------------------

    report = evaluate_fusion(
        predictions,
        labels=LABELS,
        thresholds=0.5,                 # or a {label: threshold} dict from your tuning split
        identity_labels=IDENTITY_LABELS,
    )
    print(report.render())


if __name__ == "__main__":
    main()
