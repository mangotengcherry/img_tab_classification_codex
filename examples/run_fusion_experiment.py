"""End-to-end fusion experiment: generate -> train -> predict -> evaluate -> visualize.

Run:

    PYTHONPATH=src python3 examples/run_fusion_experiment.py

Writes everything under ``reports/``:

- ``reports/figures/*.png``         data + result visualizations
- ``reports/fusion_predictions.csv`` test-set predictions (3 heads)
- ``reports/fusion_report.md`` / ``.json``  the structured evaluation
- ``reports/training_history.csv``   per-head loss per epoch

To use real data, replace ``generate_dataset()`` with your own loader that
produces the same arrays (images, tabular with NaN where unavailable, labels,
eval_group, has_tabular). See docs/fusion_eval_quickstart.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fbm_multimodal.fusion.data import generate_dataset
from fbm_multimodal.fusion.fusion_eval import evaluate_fusion, modality_contribution
from fbm_multimodal.fusion.model import FusionMLP
from fbm_multimodal.fusion.visualize import generate_all_figures

SEED = 0
REPORTS = Path("reports")
FIGURES = REPORTS / "figures"


def _features(images: np.ndarray) -> np.ndarray:
    """Normalized graded FBM, 2x2 average-pooled and flattened."""
    n, h, w = images.shape
    h2, w2 = h // 2, w // 2
    pooled = images[:, : h2 * 2, : w2 * 2].reshape(n, h2, 2, w2, 2).mean(axis=(2, 4))
    return (pooled / 8.0).reshape(n, -1)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # 1) data ---------------------------------------------------------------
    ds = generate_dataset(seed=SEED)
    labels = ds.label_names
    train = ds.split == "train"
    test = ds.split == "test"

    xi = _features(ds.images)
    xt = ds.tabular
    y = ds.labels
    has_tab = ds.has_tabular

    # 2) train --------------------------------------------------------------
    model = FusionMLP(hidden=48, lr=3e-3, epochs=300, dropout_p=0.3, seed=SEED)
    model.fit(xi[train], xt[train], y[train], has_tab[train])
    pd.DataFrame(model.history).to_csv(REPORTS / "training_history.csv", index=False)

    # 3) predict on the test set -> predictions table -----------------------
    heads = model.predict_heads(xi[test], xt[test], has_tab[test])
    pred = {"chip_id": [c for c, t in zip(ds.chip_id, test) if t], "eval_group": ds.eval_group[test]}
    for k, label in enumerate(labels):
        pred[f"true_{label}"] = y[test][:, k]
        pred[f"image_prob_{label}"] = heads["image"][:, k]
        pred[f"tabular_prob_{label}"] = heads["tabular"][:, k]
        pred[f"fusion_prob_{label}"] = heads["fusion"][:, k]
        # alias so the same CSV also feeds the core evaluate-conditions CLI
        pred[f"prob_{label}"] = heads["fusion"][:, k]
    predictions = pd.DataFrame(pred)
    predictions.to_csv(REPORTS / "fusion_predictions.csv", index=False)

    # 4) evaluate -----------------------------------------------------------
    report = evaluate_fusion(predictions, labels=labels, identity_labels=ds.identity_labels)
    (REPORTS / "fusion_report.md").write_text(report.render(), encoding="utf-8")
    (REPORTS / "fusion_report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 5) true tabular ablation on the real test rows (model-side) -----------
    real_test = test & has_tab
    ablation = modality_contribution(
        lambda images, tabular: model.predict_heads(images, tabular, np.ones(len(images), dtype=bool))["fusion"],
        xi[real_test],
        np.where(np.isnan(xt[real_test]), 0.0, xt[real_test]),
        y[real_test],
        thresholds=0.5,
    )

    # 6) visualize ----------------------------------------------------------
    figures = generate_all_figures(ds, model.history, report, FIGURES, ablation=ablation)

    # 7) console summary ----------------------------------------------------
    print(report.render())
    print("tabular ablation:", {k: round(v, 3) for k, v in ablation.items()})
    print("\nartifacts:")
    for path in [REPORTS / "fusion_predictions.csv", REPORTS / "fusion_report.md",
                 REPORTS / "fusion_report.json", REPORTS / "training_history.csv", *figures]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
