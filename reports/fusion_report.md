# Fusion evaluation report

- labels: edge_ring, center_blob, leak_top, leak_bottom
- heads present: image_only, tabular_only, fusion

## Subset accuracy by head x eval_group  (acc [CI] / support)

| head | real_composite | real_single | synthetic_composite |
|---|---|---|---|
| image_only | 0.533 [0.41,0.65] / 60 | 0.700 [0.63,0.76] / 180 | 0.607 [0.53,0.68] / 150 |
| tabular_only | 0.517 [0.39,0.64] / 60 | 1.000 [0.98,1.00] / 180 | — |
| fusion | 0.983 [0.91,1.00] / 60 | 0.972 [0.94,0.99] / 180 | — |

## KPI product per head  (single x composite, real only)

| head | single | composite | KPI product | composite support |
|---|---|---|---|---|
| image_only | 0.700 | 0.533 | 0.373 | 60 |
| tabular_only | 1.000 | 0.517 | 0.517 | 60 |
| fusion | 0.972 | 0.983 | 0.956 | 60 |

## Fusion gain

- fusion_kpi: 0.956
- best_unimodal_kpi: 0.517
- gain: 0.439
- gain_over_image_only: 0.583
- gain_over_tabular_only: 0.439

## Modality-collapse diagnostic (real composite)

- n_composite: 60.000
- image_only_subset_acc: 0.533
- fusion_subset_acc: 0.983
- fusion_gain_over_image: 0.450
- tabular_rescue_candidates: 18.000
- fusion_followed_tabular: 17.000
- fusion_follow_rate: 0.944
- fusion_regressions: 0.000

## Identity-class slice (electrical-only classes)

- identity labels: leak_top, leak_bottom
- n: 141
- image_only: acc=0.418 / support=141
- tabular_only: acc=0.865 / support=141
- fusion: acc=0.957 / support=141
- tabular_minus_image: 0.447

_No warnings._
