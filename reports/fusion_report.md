# Fusion evaluation report

- labels: edge_ring, center_blob, leak_top, leak_bottom
- heads present: image_only, tabular_only, fusion

## Subset accuracy by head x eval_group  (acc [CI] / support)

| head | real_composite | real_single | synthetic_composite |
|---|---|---|---|
| image_only | 0.517 [0.39,0.64] / 60 | 0.678 [0.61,0.74] / 180 | 0.560 [0.48,0.64] / 150 |
| tabular_only | 0.567 [0.44,0.68] / 60 | 1.000 [0.98,1.00] / 180 | — |
| fusion | 0.983 [0.91,1.00] / 60 | 0.983 [0.95,0.99] / 180 | — |

## KPI product per head  (single x composite, real only)

| head | single | composite | KPI product | composite support |
|---|---|---|---|---|
| image_only | 0.678 | 0.517 | 0.350 | 60 |
| tabular_only | 1.000 | 0.567 | 0.567 | 60 |
| fusion | 0.983 | 0.983 | 0.967 | 60 |

## Fusion gain

- fusion_kpi: 0.967
- best_unimodal_kpi: 0.567
- gain: 0.400
- gain_over_image_only: 0.617
- gain_over_tabular_only: 0.400

## Modality-collapse diagnostic (real composite)

- n_composite: 60.000
- image_only_subset_acc: 0.517
- fusion_subset_acc: 0.983
- fusion_gain_over_image: 0.467
- tabular_rescue_candidates: 19.000
- fusion_followed_tabular: 18.000
- fusion_follow_rate: 0.947
- fusion_regressions: 0.000

## Identity-class slice (electrical-only classes)

- identity labels: leak_top, leak_bottom
- n: 154
- image_only: acc=0.435 / support=154
- tabular_only: acc=0.890 / support=154
- fusion: acc=0.974 / support=154
- tabular_minus_image: 0.455

_No warnings._
